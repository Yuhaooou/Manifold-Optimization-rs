use std::fmt::Display;

use ndarray::prelude::*;
use num_traits::Float;

use super::lapack::*;

pub mod unused {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Lgesvd {
        jobu: LapackChar,
        jobvt: LapackChar,
    }

    impl Lgesvd {
        pub fn new() -> Self {
            Lgesvd {
                jobu: LapackChar::A,
                jobvt: LapackChar::A,
            }
        }

        fn check_job(job: LapackChar) {
            assert!(
                matches!(
                    job,
                    LapackChar::A | LapackChar::S | LapackChar::O | LapackChar::N
                ),
                "Invalid job for gesvd"
            )
        }

        fn check_jobs_not_both_o(&self) {
            assert!(
                (self.jobu != LapackChar::O) || (self.jobvt != LapackChar::O),
                "jobu and jobvt cannot be O at the same time for gesvd"
            );
        }

        fn job_from_str(s: &str) -> LapackChar {
            let job = LapackChar::from_str(s);
            Self::check_job(job);
            job
        }

        pub fn new_from_str(jobu_str: &str, jobvt_str: &str) -> Self {
            let jobu = Self::job_from_str(jobu_str);
            let jobvt = Self::job_from_str(jobvt_str);
            let gesvd = Lgesvd { jobu, jobvt };
            gesvd.check_jobs_not_both_o();
            gesvd
        }

        pub fn jobu_from_str(mut self, jobu_str: &str) -> Self {
            self.jobu = Self::job_from_str(jobu_str);
            self.check_jobs_not_both_o();
            self
        }

        pub fn jobvt_from_str(mut self, jobvt_str: &str) -> Self {
            self.jobvt = Self::job_from_str(jobvt_str);
            self.check_jobs_not_both_o();
            self
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Lgesdd {
        jobz: LapackChar,
    }

    impl Lgesdd {
        pub fn new() -> Self {
            Lgesdd {
                jobz: LapackChar::A,
            }
        }

        pub fn new_from_str(jobz_str: &str) -> Self {
            let jobz = LapackChar::from_str(jobz_str);
            Self::check_job(jobz);
            Lgesdd { jobz }
        }

        fn check_job(job: LapackChar) {
            assert!(
                matches!(
                    job,
                    LapackChar::A | LapackChar::S | LapackChar::O | LapackChar::N
                ),
                "Invalid job for gesdd"
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVDBackend {
    GESVD,
    GESDD,
    GESVDQ,
    GESVDX,
    GESVJ,
    GEJSV,
}

impl Display for SVDBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SVDBackend::GESVD => write!(f, "GESVD"),
            SVDBackend::GESDD => write!(f, "GESDD"),
            SVDBackend::GESVDQ => write!(f, "GESVDQ"),
            SVDBackend::GESVDX => write!(f, "GESVDX"),
            SVDBackend::GESVJ => write!(f, "GESVJ"),
            SVDBackend::GEJSV => write!(f, "GEJSV"),
        }
    }
}

impl SVDBackend {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "GESVD" => SVDBackend::GESVD,
            "GESDD" => SVDBackend::GESDD,
            "GESVDQ" => SVDBackend::GESVDQ,
            "GESVDX" => SVDBackend::GESVDX,
            "GESVJ" => SVDBackend::GESVJ,
            "GEJSV" => SVDBackend::GEJSV,
            _ => panic!("Invalid SVD backend: {}", s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVDError {
    Unconverged,
}

/// Thin Svd for F-order matrixm. This funtion will destroy the input matrix.
fn thin_svd_owned_impl<T>(
    mut mat: Array2<T>,
    order: Layout,
    backend: SVDBackend,
) -> Result<(Array2<T>, Array1<T::Real>, Array2<T>), SVDError>
where
    T: LapackElem,
{
    // For f-order mat, directly use lapack routines to compute. For c-order mat, we compute mat.t(), so m and n are reversed.
    let (m, n) = if order.is_f() {
        mat.dim()
    } else {
        (mat.dim().1, mat.dim().0)
    };
    let k = m.min(n);

    let mut vec_s = new_uninit_vec(k);
    let mut u_or_vt = new_uninit_vec(k * k);
    let (u_opt, vt_opt) = if m >= n {
        (None, Some(u_or_vt.as_mut_slice()))
    } else {
        (Some(u_or_vt.as_mut_slice()), None)
    };

    match backend {
        SVDBackend::GESVD => {
            let jobu = if m >= n { LapackChar::O } else { LapackChar::S };
            let jobvt = if m >= n { LapackChar::S } else { LapackChar::O };
            let (info, _) = T::gesvd(
                jobu,
                jobvt,
                m as i32,
                n as i32,
                mat.as_slice_memory_order_mut().unwrap(),
                m as i32,
                &mut vec_s,
                u_opt,
                m as i32,
                vt_opt,
                n as i32,
            );
            if info != 0 {
                return Err(SVDError::Unconverged);
            }
        }
        SVDBackend::GESDD => {
            let lrwork = match T::IS_REAL {
                true => 0,
                false => {
                    let mx = m.max(n) as usize;
                    let mn = m.min(n) as usize;
                    if mx > 100 * mn {
                        5 * mn * mn + 5 * mn
                    } else {
                        (5 * mn * mx + 5 * mn).max(2 * mx * mn + 2 * mn * mn + mn)
                    }
                }
            };
            let info = T::gesdd(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_slice_memory_order_mut().unwrap(),
                m as i32,
                &mut vec_s,
                u_opt,
                m as i32,
                vt_opt,
                n as i32,
                lrwork as i32,
            );
            if info != 0 {
                return Err(SVDError::Unconverged);
            }
        }
        _ => unimplemented!("SVD backend {} is not implemented yet", backend),
    }

    let res_u;
    let res_s;
    let res_vt;
    unsafe {
        res_s = Array::from_shape_vec_unchecked(k, vec_s);
        res_s.dim();

        let u_or_vt = match order {
            Layout::F => Array::from_shape_vec_unchecked((k, k).f(), u_or_vt),
            Layout::C => Array::from_shape_vec_unchecked((k, k), u_or_vt),
        };
        if m >= n {
            res_vt = u_or_vt;
            res_u = mat;
        } else {
            res_u = u_or_vt;
            res_vt = mat;
        }
    }

    match order {
        Layout::F => Ok((res_u, res_s, res_vt)),
        Layout::C => Ok((res_vt, res_s, res_u)),
    }
}

/// Thin svd for owned matrix. This funtion will destroy the input matrix.
fn thin_svd_owned<T>(
    mat: Array2<T>,
    backend: SVDBackend,
    order: Option<Layout>,
) -> Result<(Array2<T>, Array1<T::Real>, Array2<T>), SVDError>
where
    T: LapackElem,
{
    if order.is_some() {
        thin_svd_owned_impl(mat, order.unwrap(), backend)
    } else if mat.t().is_standard_layout() {
        thin_svd_owned_impl(mat, Layout::F, backend)
    } else if mat.is_standard_layout() {
        thin_svd_owned_impl(mat, Layout::C, backend)
    } else {
        // TODO
        println!("== Untested: thin_svd_r_owned with owned non-contiguous Array. ==");
        let (u, s, vt) = thin_svd_owned_impl(mat.to_owned(), Layout::C, backend)?;

        debug_assert!(
            (u.dot(&Array2::from_diag(&s).mapv(T::from_real)).dot(&vt) - mat)
                .into_iter()
                .map(T::abs)
                .reduce(T::Real::max)
                .unwrap_or(T::zero().re())
                < num_traits::Float::sqrt(T::Real::epsilon()),
            "SVD result does not reconstruct the original matrix well. This may be a bug."
        );
        Ok((u, s, vt))
    }
}

pub trait LinalgSVD {
    type Elem;
    type Real;

    fn into_svd(self) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>);

    fn into_svd_with_backend_order(
        self,
        backend: SVDBackend,
        order: Option<Layout>,
    ) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>);

    fn svd(
        &self,
        full_matrix: bool,
    ) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>);

    fn svd_with_backend_order(
        &self,
        full_matrix: bool,
        backend: SVDBackend,
        order: Option<Layout>,
    ) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>);
}

impl<T> LinalgSVD for Array2<T>
where
    T: LapackElem,
{
    type Elem = T;
    type Real = T::Real;

    /// This method will destroy the input matrix.
    fn into_svd(self) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>) {
        self.into_svd_with_backend_order(SVDBackend::GESDD, None)
    }

    fn into_svd_with_backend_order(
        self,
        backend: SVDBackend,
        order: Option<Layout>,
    ) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>) {
        thin_svd_owned(self, backend, order).unwrap()
    }

    fn svd(
        &self,
        full_matrix: bool,
    ) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>) {
        self.svd_with_backend_order(full_matrix, SVDBackend::GESDD, None)
    }

    fn svd_with_backend_order(
        &self,
        full_matrix: bool,
        backend: SVDBackend,
        order: Option<Layout>,
    ) -> (Array2<Self::Elem>, Array1<Self::Real>, Array2<Self::Elem>) {
        if full_matrix {
            unimplemented!("Full SVD is not implemented yet")
        } else {
            thin_svd_owned(self.to_owned(), backend, order).unwrap()
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(dead_code, unused)]
    use ndarray_rand::RandomExt;
    use num_complex::{Complex32 as c32, Complex64 as c64};
    use num_traits::ToPrimitive;
    use paste::item;
    use rand_distr::{Uniform, uniform::SampleUniform};

    use super::{Layout::*, SVDBackend::*, *};

    const M: usize = 5;
    const N: usize = 4;
    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-5;

    fn test_svd_inner<T>(mat: Array2<T>, backend: SVDBackend, order: Layout, eps: T::Real)
    where
        T: LapackElem,
    {
        let (u, s, vt) = mat.svd_with_backend_order(false, backend, Some(order));
        let s = s.mapv(T::from_real);
        let a_re = u.dot(&Array2::from_diag(&s)).dot(&vt);
        let err = (a_re - mat)
            .into_iter()
            .map(T::abs)
            .reduce(T::Real::max)
            .unwrap_or(T::zero().re());
        assert!(
            err < eps,
            "SVD decomposition is not accurate enough: err = {}",
            err.to_f64().unwrap()
        );
    }

    fn test_svd<T>(m: usize, n: usize, order: Layout, backend: SVDBackend, eps: T::Real)
    where
        T: LapackElem,
        T::Real: SampleUniform,
    {
        let sh = match order {
            F => (m, n).f(),
            C => (m, n).into_shape_with_order(),
        };
        let a = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let b = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let c = a.mapv(T::from_real) + b.mapv(|x| T::from_real(x).mul_i_or_one());
        test_svd_inner(c, backend, order, eps);
    }

    macro_rules! svd_test_impl {
        ($name:tt, $ty:ty, $m:expr, $n:expr, $order:expr, $backend:expr, $eps:expr) => {
            item! {
                #[test]
                fn $name() {
                    test_svd::<$ty>($m, $n, $order, $backend, $eps);
                }
            }
        };
    }

    macro_rules! svd_inner_tests {
        ($ty_name:ident, $ty:ty, $backend_name:ident, $backend:expr, $eps:expr) => {
            svd_test_impl!([<test_f_mn_ $backend_name _ $ty_name>], $ty, M, N, F, $backend, $eps);
            svd_test_impl!([<test_f_nm_ $backend_name _ $ty_name>], $ty, N, M, F, $backend, $eps);
            svd_test_impl!([<test_c_mn_ $backend_name _ $ty_name>], $ty, M, N, C, $backend, $eps);
            svd_test_impl!([<test_c_nm_ $backend_name _ $ty_name>], $ty, N, M, C, $backend, $eps);
        };
    }

    macro_rules! svd_backend_tests {
        ($ty_name:ident, $ty:ty, $eps:expr) => {
            svd_inner_tests!($ty_name, $ty, gesvd, GESVD, $eps);
            svd_inner_tests!($ty_name, $ty, gesdd, GESDD, $eps);
        };
    }

    macro_rules! svd_test {
        ($ty_name:ident, $ty:ty, $eps:expr) => {
            svd_backend_tests!($ty_name, $ty, $eps);
        };
    }

    svd_test!(f64, f64, EPS_F64);
    svd_test!(f32, f32, EPS_F32);
    svd_test!(c64, c64, EPS_F64);
    svd_test!(c32, c32, EPS_F32);
}
