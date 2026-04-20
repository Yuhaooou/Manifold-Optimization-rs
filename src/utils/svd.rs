use std::fmt::Display;

use ndarray::prelude::*;
use num_traits::Float;

use crate::utils::lapack::*;

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

/// Thin Svd for F-order matrixm. This funtion will destroy the input matrix.
fn thin_svd_owned_impl<T: LapackRoutines>(
    mut mat: Array2<T>,
    order: Layout,
    backend: SVDBackend,
) -> (Array2<T>, Array1<T::Real>, Array2<T>) {
    // For f-order mat, directly use lapack routines to compute. For c-order mat, we compute mat.t(), so m and n are reversed.
    let (m, n) = if order.is_f() {
        mat.dim()
    } else {
        (mat.dim().1, mat.dim().0)
    };
    let r = m.min(n);
    let mut vec_s = new_uninit_vec(r);
    let mut vec_u_or_vt = new_uninit_vec(r * r);
    let u_pointer = if m >= n {
        None
    } else {
        Some(vec_u_or_vt.as_mut_ptr())
    };
    let vt_pointer: Option<*mut T> = if m >= n {
        Some(vec_u_or_vt.as_mut_ptr())
    } else {
        None
    };

    match backend {
        SVDBackend::GESVD => {
            let jobu = if m >= n { LapackChar::O } else { LapackChar::S };
            let jobvt = if m >= n { LapackChar::S } else { LapackChar::O };
            let mut work = [T::zero()];
            let (info1, _) = T::gesvd(
                jobu,
                jobvt,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                u_pointer,
                m as i32,
                vt_pointer,
                n as i32,
                work.as_mut_ptr(),
                -1_i32,
            );
            if info1 != 0 {
                println!("gesvd returned non-zero info: {}", info1);
            }
            let work_len = work[0];
            let mut work = Vec::with_capacity(work_len.to_usize().unwrap());
            let (info2, _) = T::gesvd(
                jobu,
                jobvt,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                u_pointer,
                m as i32,
                vt_pointer,
                n as i32,
                work.as_mut_ptr(),
                work_len.to_i32().unwrap(),
            );
            if info2 != 0 {
                println!("gesvd returned non-zero info: {}", info2);
            }
        }
        SVDBackend::GESDD => {
            let mut work = [T::zero()];
            let (info1, _) = T::gesdd(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                u_pointer,
                m as i32,
                vt_pointer,
                n as i32,
                work.as_mut_ptr(),
                None,
                -1_i32,
            );
            if info1 != 0 {
                println!("gesdd returned non-zero info: {}", info1);
            }
            let work_len = work[0];
            let mut work = Vec::with_capacity(work_len.to_usize().unwrap());
            // See Lapack gesdd documentation for details.
            let mut rwork = match T::IS_REAL {
                true => None,
                false => Some({
                    let mx = m.max(n) as usize;
                    let mn = m.min(n) as usize;
                    let rwork_len = if mx > 100 * mn {
                        5 * mn * mn + 5 * mn
                    } else {
                        (5 * mn * mx + 5 * mn).max(2 * mx * mn + 2 * mn * mn + mn)
                    };
                    new_uninit_vec(rwork_len)
                }),
            };
            let (info2, _) = T::gesdd(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                u_pointer,
                m as i32,
                vt_pointer,
                n as i32,
                work.as_mut_ptr(),
                if T::IS_REAL {
                    None
                } else {
                    Some(rwork.as_mut().unwrap().as_mut_ptr())
                },
                work_len.to_i32().unwrap(),
            );
            if info2 != 0 {
                println!("gesdd returned non-zero info: {}", info2);
            }
        }
        _ => unimplemented!("SVD backend {} is not implemented yet", backend),
    }

    let res_u;
    let res_s;
    let res_vt;
    unsafe {
        res_s = Array::from_shape_vec_unchecked(r, vec_s);
        res_s.dim();

        let u_or_vt = match order {
            Layout::F => Array::from_shape_vec_unchecked((r, r).f(), vec_u_or_vt),
            Layout::C => Array::from_shape_vec_unchecked((r, r), vec_u_or_vt),
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
        Layout::F => (res_u, res_s, res_vt),
        Layout::C => (res_vt, res_s, res_u),
    }
}

/// Thin svd for owned matrix. This funtion will destroy the input matrix.
pub fn thin_svd_owned<T: LapackRoutines>(
    mat: Array2<T>,
    backend: SVDBackend,
    order: Option<Layout>,
) -> (Array2<T>, Array1<T::Real>, Array2<T>) {
    if order.is_some() {
        thin_svd_owned_impl(mat, order.unwrap(), backend)
    } else if mat.t().is_standard_layout() {
        thin_svd_owned_impl(mat, Layout::F, backend)
    } else if mat.is_standard_layout() {
        thin_svd_owned_impl(mat, Layout::C, backend)
    } else {
        // TODO
        println!("== Untested: thin_svd_r_owned with owned non-contiguous Array. ==");
        let (u, s, vt) = thin_svd_owned_impl(mat.to_owned(), Layout::C, backend);

        debug_assert!(
            (u.dot(&Array2::from_diag(&s).mapv(T::from_real)).dot(&vt) - mat)
                .into_iter()
                .map(T::abs)
                .reduce(T::Real::max)
                .unwrap_or(T::zero().re())
                < T::Real::epsilon().sqrt(),
            "SVD result does not reconstruct the original matrix well. This may be a bug."
        );
        (u, s, vt)
    }
}

pub fn thin_svd_ref<T: LapackRoutines>(
    mat: &Array2<T>,
    backend: SVDBackend,
    order: Option<Layout>,
) -> (Array2<T>, Array1<T::Real>, Array2<T>) {
    thin_svd_owned(mat.to_owned(), backend, order)
}

#[cfg(test)]
mod test {
    #![allow(dead_code, unused)]
    use ndarray_rand::RandomExt;
    use num_complex::{Complex32 as c32, Complex64 as c64};
    use paste::item;
    use rand_distr::{Uniform, uniform::SampleUniform};

    use super::{Layout::*, SVDBackend::*, *};

    const M: usize = 5;
    const N: usize = 4;
    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-5;

    fn test_svd_inner<T>(mat: Array2<T>, backend: SVDBackend, order: Layout, eps: T::Real)
    where
        T: LapackRoutines,
    {
        let (u, s, vt) = thin_svd_owned(mat.clone(), backend, Some(order));
        let s = s.mapv(T::from_real);
        let a_re = u.dot(&Array2::from_diag(&s)).dot(&vt);
        let err = (a_re - mat)
            .into_iter()
            .map(T::abs)
            .reduce(T::Real::max)
            .unwrap_or(T::zero().re());
        assert!(err < eps);
    }

    fn test_svd<T>(m: usize, n: usize, order: Layout, backend: SVDBackend, eps: T::Real)
    where
        T: LapackRoutines,
        T::Real: SampleUniform,
    {
        let sh = match order {
            F => (m, n).f(),
            C => (m, n).into_shape_with_order(),
        };
        let a = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let b = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let c = a.mapv(T::from_real) + b.mapv(|x| T::from_real(x) * T::get_i());
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
