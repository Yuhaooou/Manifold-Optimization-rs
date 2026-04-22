use ndarray::prelude::*;

use crate::linalg::lapack::{LapackElem, Layout, new_uninit_vec};

pub enum QRBackend {
    GEQRF,
    GEQRFP,
}

fn qr_owned_impl<T>(mut mat: Array2<T>, order: Layout, backend: QRBackend) -> (Array2<T>, Array2<T>)
where
    T: LapackElem,
{
    let (m, n) = match order {
        Layout::F => mat.dim(),
        Layout::C => (mat.dim().1, mat.dim().0),
    };
    let k = m.min(n);

    let geqrf = match backend {
        QRBackend::GEQRF => T::geqrf,
        QRBackend::GEQRFP => T::geqrfp,
    };

    // Haouseholder
    let mut tau = new_uninit_vec(k);
    let mut work = [T::zero()];
    let info1 = geqrf(
        m as i32,
        n as i32,
        mat.as_slice_memory_order_mut().unwrap(),
        m as i32,
        tau.as_mut_slice(),
        &mut work,
        -1,
    );
    if info1 != 0 {
        panic!("Error in geqrf: {}", info1);
    }
    let lwork = work[0].to_usize().unwrap();
    let mut work = new_uninit_vec(lwork);
    let info2 = geqrf(
        m as i32,
        n as i32,
        mat.as_slice_memory_order_mut().unwrap(),
        m as i32,
        tau.as_mut_slice(),
        &mut work,
        lwork as i32,
    );
    if info2 != 0 {
        panic!("Error in geqrf: {}", info2);
    }

    let mat_r = if m > n {
        mat.slice(s![..n, ..]).to_owned().triu(0)
    } else {
        mat.triu(0)
    };

    // reconstruct q
    let mut work = [T::zero()];
    let info1 = T::qfrom(
        m as i32,
        k as i32,
        k as i32,
        mat.as_slice_memory_order_mut().unwrap(),
        m as i32,
        tau.as_mut_slice(),
        &mut work,
        -1,
    );
    if info1 != 0 {
        panic!("Error in qfrom: {}", info1);
    }
    let lwork = work[0].to_usize().unwrap();
    let mut work = new_uninit_vec(lwork);
    let info2 = T::qfrom(
        m as i32,
        k as i32,
        k as i32,
        mat.as_slice_memory_order_mut().unwrap(),
        m as i32,
        tau.as_mut_slice(),
        work.as_mut_slice(),
        lwork as i32,
    );
    if info2 != 0 {
        panic!("Error in qfrom: {}", info2);
    }

    let mat_q = if k < n {
        mat.slice(s![.., ..k]).to_owned()
    } else {
        mat
    };

    (mat_q, mat_r)
}

pub trait LinalgQR {
    type Elem;

    fn into_qr(self) -> (Array2<Self::Elem>, Array2<Self::Elem>);

    fn into_qr_with_backend(self, backend: QRBackend) -> (Array2<Self::Elem>, Array2<Self::Elem>);

    fn qr(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.qr_with_backend(QRBackend::GEQRF)
    }

    fn qr_unique(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.qr_with_backend(QRBackend::GEQRFP)
    }

    fn qr_with_backend(&self, backend: QRBackend) -> (Array2<Self::Elem>, Array2<Self::Elem>);
}

impl<T> LinalgQR for Array2<T>
where
    T: LapackElem,
{
    type Elem = T;

    fn into_qr(self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.into_qr_with_backend(QRBackend::GEQRF)
    }

    fn into_qr_with_backend(self, backend: QRBackend) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        if self.t().is_standard_layout() {
            qr_owned_impl(self, Layout::F, backend)
        } else if self.is_standard_layout() {
            let mut mat_f = Array2::from_elem(self.raw_dim().f(), T::zero());
            mat_f.assign(&self);
            qr_owned_impl(mat_f, Layout::F, backend)
        } else {
            panic!("Matrix must be in standard layout or its transpose must be in standard layout");
        }
    }

    fn qr_with_backend(&self, backend: QRBackend) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.to_owned().into_qr_with_backend(backend)
    }
}

#[cfg(test)]
mod tests {
    #![allow(dead_code, unused)]
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use num_complex::{Complex32 as c32, Complex64 as c64};
    use num_traits::Float;
    use num_traits::ToPrimitive;
    use paste::item;
    use rand_distr::{Uniform, uniform::SampleUniform};

    use super::{*, QRBackend::*};
    use crate::linalg::lapack::Layout::{C, F};
    use crate::utils::traits::RCLike;

    const M: usize = 5;
    const N: usize = 4;
    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-5;

    fn test_qr_inner<T>(mat: Array2<T>, backend: QRBackend, eps: T::Real)
    where
        T: RCLike + LapackElem,
    {
        let (q, r) = mat.qr_with_backend(backend);
        let qtq = q.t().mapv(|x| x.conj()).dot(&q);
        let err_q = (&qtq - Array2::eye(qtq.dim().0).mapv(T::from_real))
            .into_iter()
            .map(T::abs)
            .reduce(T::Real::max)
            .unwrap_or(T::zero().re());
        assert!(
            err_q < eps,
            "Q is not orthogonal enough: err_q = {}",
            err_q.to_f64().unwrap()
        );

        let rec = q.dot(&r);
        let err = (&rec - &mat)
            .into_iter()
            .map(T::abs)
            .reduce(T::Real::max)
            .unwrap_or(T::zero().re());
        assert!(
            err < eps,
            "QR decomposition is not accurate enough: err = {}",
            err.to_f64().unwrap()
        );
    }

    fn test_qr<T>(m: usize, n: usize, order: Layout, backend: QRBackend, eps: T::Real)
    where
        T: RCLike + LapackElem,
        T::Real: SampleUniform,
    {
        let sh = match order {
            F => (m, n).f(),
            C => (m, n).into_shape_with_order(),
        };
        let a = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let b = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let c = a.mapv(T::from_real) + b.mapv(|x| T::from_real(x).mul_i_or_one());
        test_qr_inner(c, backend, eps);
    }

    macro_rules! qr_test_impl {
        ($name:tt, $ty:ty, $m:expr, $n:expr, $order:expr, $backend:expr, $eps:expr) => {
            item! {
                #[test]
                fn $name() {
                    test_qr::<$ty>($m, $n, $order, $backend, $eps);
                }
            }
        };
    }

    macro_rules! qr_inner_tests {
        ($ty_name:ident, $ty:ty, $backend_name:ident, $backend:expr, $eps:expr) => {
            qr_test_impl!([<test_f_mn_qr_ $backend_name _ $ty_name>], $ty, M, N, F, $backend, $eps);
            qr_test_impl!([<test_f_nm_qr_ $backend_name _ $ty_name>], $ty, N, M, F, $backend, $eps);
            qr_test_impl!([<test_c_mn_qr_ $backend_name _ $ty_name>], $ty, M, N, C, $backend, $eps);
            qr_test_impl!([<test_c_nm_qr_ $backend_name _ $ty_name>], $ty, N, M, C, $backend, $eps);
        };
    }

    macro_rules! qr_backend_tests {
        ($ty_name:ident, $ty:ty, $eps:expr) => {
            qr_inner_tests!($ty_name, $ty, gesvd, GEQRF, $eps);
            qr_inner_tests!($ty_name, $ty, gesdd, GEQRFP, $eps);
        };
    }

    macro_rules! qr_test {
        ($ty_name:ident, $ty:ty, $eps:expr) => {
            qr_backend_tests!($ty_name, $ty, $eps);
        };
    }

    qr_test!(f64, f64, EPS_F64);
    qr_test!(f32, f32, EPS_F32);
    qr_test!(c64, c64, EPS_F64);
    qr_test!(c32, c32, EPS_F32);
}
