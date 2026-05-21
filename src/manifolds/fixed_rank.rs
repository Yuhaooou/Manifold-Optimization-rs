use std::ops::{Add, Div, Mul, Neg, Sub};

use ndarray::{ScalarOperand, concatenate, prelude::*};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::linalg::{LapackElem, LinalgQR, LinalgSVD};
use crate::manifolds::{EGradToRGrad, EHessToRHess, Manifold, RandomPoint};
use crate::random_point_forward;
use crate::utils::traits::{InnerProduct, RCLike, Vector};

#[derive(Debug, Clone)]
pub struct FixedRank<D>
where
    D: RCLike,
{
    name: String,
    m: usize,
    n: usize,
    r: usize,
    _marker: std::marker::PhantomData<D>,
}

impl<D> FixedRank<D>
where
    D: RCLike + ScalarOperand,
{
    pub fn new(m: usize, n: usize, r: usize) -> Self {
        assert!(
            m >= r && n >= r && r >= 1,
            "Need m >= r >= 1 and n >= r >= 1"
        );
        FixedRank {
            name: format!(
                "Fixed-rank manifold of rank-{} matrices in R^({}, {})",
                r, m, n
            ),
            m,
            n,
            r,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn set_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

impl<D> Manifold for FixedRank<D>
where
    D: RCLike + ScalarOperand + LapackElem,
{
    type Point = Point<D>;
    type TangentVector = TangentVector<D>;
    type AmbientPoint = Array2<D>;
    type Field = D;

    fn base_point(&self) -> Self::Point {
        let u = Array2::eye(self.m).slice(s![.., ..self.r]).to_owned();
        let s = Array1::ones(self.r);
        let v = Array2::eye(self.n).slice(s![.., ..self.r]).to_owned();
        Point { u, s, v }
    }

    fn zero_tangent_vector(&self, _point: &Self::Point) -> Self::TangentVector {
        let up = Array2::zeros((self.m, self.r));
        let m = Array2::zeros((self.r, self.r));
        let vp = Array2::zeros((self.n, self.r));
        TangentVector { up, m, vp }
    }

    fn inner(
        &self,
        _point: &Self::Point,
        tangent_vector1: &Self::TangentVector,
        tangent_vector2: &Self::TangentVector,
    ) -> D::Real {
        let (up1, m1, vp1) = (&tangent_vector1.up, &tangent_vector1.m, &tangent_vector1.vp);
        let (up2, m2, vp2) = (&tangent_vector2.up, &tangent_vector2.m, &tangent_vector2.vp);
        up1.inner(up2) + m1.inner(m2) + vp1.inner(vp2)
    }

    fn retraction(&self, point: &Self::Point, tangent_vector: &Self::TangentVector) -> Self::Point {
        let uup = concatenate![Axis(1), point.u, tangent_vector.up];
        let (qu, ru) = uup.into_qr();
        let vvp = concatenate![Axis(1), point.v, tangent_vector.vp];
        let (qv, rv) = vvp.into_qr();
        let (svdu, svds, svdv) = {
            let tmp = &tangent_vector.m + Array2::from_diag(&point.s);
            let tmp = concatenate![Axis(1), tmp, Array2::eye(self.r)];
            let tmp2 = concatenate![
                Axis(1),
                Array2::eye(self.r),
                Array2::zeros((self.r, self.r))
            ];
            let tmp = concatenate![Axis(0), tmp, tmp2];
            let tmpsvd = (ru.dot(&tmp).dot(&rv.t())).into_svd();
            (
                // TOOD: Directly use truncated SVD.
                tmpsvd.0.slice(s![.., ..self.r]).to_owned(),
                tmpsvd.1.slice(s![..self.r]).to_owned(),
                tmpsvd.2.slice(s![..self.r, ..]).t().to_owned(),
            )
        };

        Self::Point::new(qu.dot(&svdu), svds.mapv(RCLike::from_real), qv.dot(&svdv))
    }

    fn projection(
        &self,
        point: &Self::Point,
        ambient_vector: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        let tmp1 = ambient_vector.dot(&point.v);
        let m = point.u.t().dot(&tmp1);
        let up = tmp1 - point.u.dot(&m);
        let vp = ambient_vector.t().dot(&point.u) - point.v.dot(&m.t());
        TangentVector { up, m, vp }
    }
}

impl<D> EGradToRGrad for FixedRank<D>
where
    D: RCLike + ScalarOperand + LapackElem,
{
    fn egrad_to_rgrad(
        &self,
        point: &Self::Point,
        egrad: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        self.projection(point, &egrad)
    }
}

impl<D> EHessToRHess for FixedRank<D>
where
    D: RCLike + ScalarOperand + LapackElem,
{
    fn ehess_to_rhess(
        &self,
        point: &Self::Point,
        tangent_vector: &Self::TangentVector,
        egrad: &Self::AmbientPoint,
        ehess: &Self::AmbientPoint,
    ) -> Self::TangentVector {
        let pu_c = Array2::eye(self.m) - point.u.dot(&point.u.t());
        let pv_c = Array2::eye(self.n) - point.v.dot(&point.v.t());
        let s_inv = point.s.mapv(|x| {
            if x != D::zero() {
                D::one() / x
            } else {
                D::zero()
            }
        });
        let m = point.u.t().dot(ehess).dot(&point.v);
        let tmp = ehess.dot(&point.v) + egrad.dot(&tangent_vector.vp) * &s_inv;
        let up = pu_c.dot(&tmp);
        let tmp = ehess.t().dot(&point.u) + egrad.t().dot(&tangent_vector.up) * &s_inv;
        let vp = pv_c.dot(&tmp);
        TangentVector { up, m, vp }
    }
}

impl<D> RandomPoint for FixedRank<D>
where
    D: RCLike + ScalarOperand + LapackElem,
    StandardNormal: Distribution<D::Real>,
{
    random_point_forward!(StandardNormal);

    fn random_point_impl<Dist, R>(&self, dist: Dist, rng: &mut R) -> Self::Point
    where
        Dist: rand::prelude::Distribution<<Self::Field as num_complex::ComplexFloat>::Real>,
        R: Rng + ?Sized,
    {
        let u = Array2::random_using((self.m, self.r), &dist, rng).mapv(D::from_real);
        let v = Array2::random_using((self.n, self.r), &dist, rng).mapv(D::from_real);

        let full = u.dot(&v.t());
        Self::Point::new_from_full(&full, self.r)
    }
}

#[derive(Debug, Clone)]
pub struct Point<D> {
    u: Array2<D>,
    s: Array1<D>,
    v: Array2<D>,
}

impl<D> Point<D> {
    pub fn new(u: Array2<D>, s: Array1<D>, v: Array2<D>) -> Self {
        Point { u, s, v }
    }

    pub fn u(&self) -> &Array2<D> {
        &self.u
    }

    pub fn s(&self) -> &Array1<D> {
        &self.s
    }

    pub fn v(&self) -> &Array2<D> {
        &self.v
    }

    pub fn vt<'a>(&'a self) -> ArrayView2<'a, D> {
        self.v.t()
    }

    pub fn full(&self) -> Array2<D>
    where
        D: RCLike + ScalarOperand,
    {
        (&self.u * &self.s).dot(&self.v.t())
    }

    pub fn new_from_full(full: &Array2<D>, r: usize) -> Self
    where
        D: RCLike + ScalarOperand + LapackElem,
    {
        let (u, s, v) = full.svd(false);
        Point::new(
            u.slice(s![.., ..r]).to_owned(),
            s.slice(s![..r]).to_owned().mapv(RCLike::from_real),
            v.slice(s![.., ..r]).to_owned(),
        )
    }
}

impl std::fmt::Display for Point<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Point {{ u: {:.6e}, s: {:.6e}, v: {:.6e} }}",
            self.u, self.s, self.v
        )
    }
}

#[derive(Debug, Clone)]
pub struct TangentVector<D> {
    up: Array2<D>,
    m: Array2<D>,
    vp: Array2<D>,
}

impl<D> TangentVector<D> {
    pub fn new(up: Array2<D>, m: Array2<D>, vp: Array2<D>) -> Self {
        TangentVector { up, m, vp }
    }

    pub fn up(&self) -> &Array2<D> {
        &self.up
    }

    pub fn m(&self) -> &Array2<D> {
        &self.m
    }

    pub fn vp(&self) -> &Array2<D> {
        &self.vp
    }

    pub fn full(&self, point: &Point<D>) -> Array2<D>
    where
        D: RCLike + ScalarOperand,
    {
        point.u.dot(&self.m).dot(&point.v.t())
            + self.up.dot(&point.v.t())
            + point.u.dot(&self.vp.t())
    }
}

impl<D> Add for TangentVector<D>
where
    D: RCLike,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.up + other.up, self.m + other.m, self.vp + other.vp)
    }
}

impl<'a, D> Add<&'a TangentVector<D>> for TangentVector<D>
where
    D: RCLike,
{
    type Output = TangentVector<D>;

    fn add(self, other: &TangentVector<D>) -> TangentVector<D> {
        TangentVector::new(self.up + &other.up, self.m + &other.m, self.vp + &other.vp)
    }
}

impl<D> Sub for TangentVector<D>
where
    D: RCLike,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.up - other.up, self.m - other.m, self.vp - other.vp)
    }
}

impl<'a, D> Sub<&'a TangentVector<D>> for TangentVector<D>
where
    D: RCLike,
{
    type Output = TangentVector<D>;

    fn sub(self, other: &TangentVector<D>) -> TangentVector<D> {
        TangentVector::new(self.up - &other.up, self.m - &other.m, self.vp - &other.vp)
    }
}

impl<D> Mul<D> for TangentVector<D>
where
    D: RCLike + ScalarOperand,
{
    type Output = Self;

    fn mul(self, num: D) -> Self {
        Self::new(self.up * num, self.m * num, self.vp * num)
    }
}

impl<D> Div<D> for TangentVector<D>
where
    D: RCLike + ScalarOperand,
{
    type Output = Self;

    fn div(self, num: D) -> Self {
        Self::new(self.up / num, self.m / num, self.vp / num)
    }
}

impl<D> Neg for TangentVector<D>
where
    D: RCLike + ScalarOperand,
{
    type Output = Self;

    fn neg(self) -> Self {
        TangentVector {
            up: -self.up,
            m: -self.m,
            vp: -self.vp,
        }
    }
}

impl<D> Vector for TangentVector<D>
where
    D: RCLike + ScalarOperand,
{
    type Field = D;

    fn ref_mul_num(&self, num: Self::Field) -> Self {
        Self::new(&self.up * num, &self.m * num, &self.vp * num)
    }

    fn ref_div_num(&self, num: Self::Field) -> Self {
        Self::new(&self.up / num, &self.m / num, &self.vp / num)
    }

    fn ref_add(&self, rhs: Self) -> Self {
        Self::new(&self.up + rhs.up, &self.m + rhs.m, &self.vp + rhs.vp)
    }

    fn ref_sub(&self, rhs: Self) -> Self {
        Self::new(&self.up - rhs.up, &self.m - rhs.m, &self.vp - rhs.vp)
    }

    fn ref_neg(&self) -> Self {
        Self::new(-self.up.clone(), -self.m.clone(), -self.vp.clone())
    }

    fn ref_add_ref(&self, rhs: &Self) -> Self {
        Self::new(&self.up + &rhs.up, &self.m + &rhs.m, &self.vp + &rhs.vp)
    }

    fn ref_sub_ref(&self, rhs: &Self) -> Self {
        Self::new(&self.up - &rhs.up, &self.m - &rhs.m, &self.vp - &rhs.vp)
    }

    fn ref_add_num(&self, num: Self::Field) -> Self {
        Self::new(&self.up + num, &self.m + num, &self.vp + num)
    }

    fn ref_sub_num(&self, num: Self::Field) -> Self {
        Self::new(&self.up - num, &self.m - num, &self.vp - num)
    }

    fn elementwise_mul(&self, _rhs: &Self) -> Self {
        unimplemented!("avoid elementwise mul on this manifold's tangent space")
    }

    fn elementwise_div(&self, _rhs: &Self) -> Self {
        unimplemented!("avoid elementwise div on this manifold's tangent space")
    }

    fn zeros_like(&self) -> Self {
        Self::new(
            Array2::zeros(self.up.dim()),
            Array2::zeros(self.m.dim()),
            Array2::zeros(self.vp.dim()),
        )
    }

    fn nums_like(&self, _num: Self::Field) -> Self {
        unimplemented!("avoid creating tangent vectors from numbers on this manifold")
    }
}

#[allow(unused_imports)]
mod tests {
    use crate::utils::traits::Norm;

    use super::*;
    use rand::distr::Uniform;

    #[test]
    fn test_stiefel() {
        let m = 50;
        let n = 40;
        let r = 5;
        let manifold = FixedRank::<f64>::new(m, n, r);

        let point = manifold.random_point();
        let _full_point = point.full();
        let ambient_point = Array2::random((m, n), Uniform::new(0., 1.).unwrap());

        let tangent_vector = manifold.projection(&point, &ambient_point);
        let _full_tangent_vector = tangent_vector.full(&point);

        let _retracted_point = manifold.retraction(&point, &(tangent_vector * 1.2));
    }
}
