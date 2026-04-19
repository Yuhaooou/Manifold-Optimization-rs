use ndarray::prelude::*;
use num_traits::{Float, NumCast};

use crate::utils::lapack::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVDBackend {
    GESVD,
    GESDD,
    GESVDQ,
    GESVDX,
    GESVJ,
    GEJSV,
}

impl SVDBackend {}

pub trait LinalgSVD {}

/// Thin Svd for F-order matrixm. This funtion will destroy the input matrix.
fn thin_svd_r_owned_f<T: LapackElem + NumCast>(
    mut mat: Array2<T>,
    backend: SVDBackend,
) -> (Array2<T>, Array1<T>, Array2<T>) {
    let (m, n) = mat.dim();
    let r = m.min(n);
    let res_u;
    let res_s;
    let res_vt;
    let mut vec_s = Vec::with_capacity(r);
    let mut vec_u_or_vt = Vec::with_capacity(r * r);
    unsafe {
        vec_s.set_len(r);
        vec_u_or_vt.set_len(r * r);
    }
    let vec_u = if m >= n {
        None
    } else {
        Some(vec_u_or_vt.as_mut_ptr())
    };
    let vec_vt = if m >= n {
        Some(vec_u_or_vt.as_mut_ptr())
    } else {
        None
    };

    match backend {
        SVDBackend::GESVD => {
            let jobu = if m >= n { LapackChar::O } else { LapackChar::S };
            let jobvt = if m >= n { LapackChar::S } else { LapackChar::O };
            let mut work = [T::default()];
            let info1 = gesvd_r(
                jobu,
                jobvt,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                vec_u,
                m as i32,
                vec_vt,
                n as i32,
                work.as_mut_ptr(),
                -1_i32,
            );
            if info1 != 0 {
                println!("gesvd_r returned non-zero info: {}", info1);
            }
            let work_len = work[0];
            let mut work = Vec::with_capacity(work_len.to_usize().unwrap());
            let info2 = gesvd_r(
                jobu,
                jobvt,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                vec_u,
                m as i32,
                vec_vt,
                n as i32,
                work.as_mut_ptr(),
                work_len.to_i32().unwrap(),
            );
            if info2 != 0 {
                println!("gesvd_r returned non-zero info: {}", info2);
            }
        }
        SVDBackend::GESDD => {
            let mut vec_i = Vec::with_capacity(8 * r);
            unsafe {
                vec_i.set_len(8 * r);
            }

            let mut work = [T::default()];
            let info1 = gesdd_r(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                vec_u,
                m as i32,
                vec_vt,
                n as i32,
                work.as_mut_ptr(),
                -1_i32,
                vec_i.as_mut_ptr(),
            );
            if info1 != 0 {
                println!("gesvd_r returned non-zero info: {}", info1);
            }
            let work_len = work[0];
            let mut work = Vec::with_capacity(work_len.to_usize().unwrap());
            let info2 = gesdd_r(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                vec_s.as_mut_ptr(),
                vec_u,
                m as i32,
                vec_vt,
                n as i32,
                work.as_mut_ptr(),
                work_len.to_i32().unwrap(),
                vec_i.as_mut_ptr(),
            );
            if info2 != 0 {
                println!("gesvd_r returned non-zero info: {}", info2);
            }
        }
        _ => unimplemented!("SVD backend {:?} is not implemented yet", backend),
    }
    unsafe {
        res_s = Array::from_shape_vec_unchecked(r, vec_s);
        let u_or_vt = Array::from_shape_vec_unchecked((r, r).f(), vec_u_or_vt);
        if m >= n {
            res_vt = u_or_vt;
            res_u = mat;
        } else {
            res_u = u_or_vt;
            res_vt = mat;
        }
    }
    (res_u, res_s, res_vt)
}

/// Thin Svd for C-order matrix. This funtion will destroy the input matrix.
fn thin_svd_r_owned_c<T: LapackElem + NumCast>(
    mat: Array2<T>,
    backend: SVDBackend,
) -> (Array2<T>, Array1<T>, Array2<T>) {
    let (ut, s, vtt) = thin_svd_r_owned_f(mat.reversed_axes(), backend);
    (vtt.reversed_axes(), s, ut.reversed_axes())
}

/// Thin svd for owned matrix. This funtion will destroy the input matrix.
pub fn thin_svd_r_owned<T: LapackElem + NumCast>(
    mat: Array2<T>,
    backend: SVDBackend,
) -> (Array2<T>, Array1<T>, Array2<T>) {
    if mat.t().is_standard_layout() {
        thin_svd_r_owned_f(mat, backend)
    } else if mat.is_standard_layout() {
        thin_svd_r_owned_c(mat, backend)
    } else {
        unreachable!("Input matrix must be either C-order or F-order"); //?
    }
}

pub mod test {
    #![allow(dead_code, unused)]
    use ndarray_rand::RandomExt;
    use rand_distr::{Uniform, uniform::SampleUniform};

    use super::{MatrixOrder::*, SVDBackend::*, *};

    const M: usize = 5;
    const N: usize = 4;
    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-5;

    fn test_svd<T>(m: usize, n: usize, order: MatrixOrder, backend: SVDBackend, eps: T)
    where
        T: Float + SampleUniform + LapackElem + std::fmt::Debug,
    {
        let sh = if order == F {
            (m, n).f()
        } else {
            (m, n).into_shape_with_order()
        };
        let a: Array2<T> = Array::random(sh, Uniform::new(T::zero(), T::one()).unwrap());
        let (u, s, vt) = thin_svd_r_owned(a.clone(), backend);
        let a_re = u.dot(&Array2::from_diag(&s)).dot(&vt);
        let err = (a_re - a)
            .abs()
            .into_iter()
            .reduce(T::max)
            .unwrap_or(T::zero());
        assert!(err < eps);
    }

    macro_rules! svd_test {
        ($name:ident, $ty:ty, $m:expr, $n:expr, $order:expr, $backend:expr, $eps:expr) => {
            #[test]
            fn $name() {
                test_svd::<$ty>($m, $n, $order, $backend, $eps);
            }
        };
    }

    svd_test!(test_f64_gesvd_f_mn, f64, M, N, F, GESVD, EPS_F64);
    svd_test!(test_f64_gesvd_f_nm, f64, N, M, F, GESVD, EPS_F64);
    svd_test!(test_f64_gesvd_c_mn, f64, M, N, C, GESVD, EPS_F64);
    svd_test!(test_f64_gesvd_c_nm, f64, N, M, C, GESVD, EPS_F64);
    svd_test!(test_f64_gesdd_f_mn, f64, M, N, F, GESDD, EPS_F64);
    svd_test!(test_f64_gesdd_f_nm, f64, N, M, F, GESDD, EPS_F64);
    svd_test!(test_f64_gesdd_c_mn, f64, M, N, C, GESDD, EPS_F64);
    svd_test!(test_f64_gesdd_c_nm, f64, N, M, C, GESDD, EPS_F64);
    svd_test!(test_f32_gesvd_f_mn, f32, M, N, F, GESVD, EPS_F32);
    svd_test!(test_f32_gesvd_f_nm, f32, N, M, F, GESVD, EPS_F32);
    svd_test!(test_f32_gesvd_c_mn, f32, M, N, C, GESVD, EPS_F32);
    svd_test!(test_f32_gesvd_c_nm, f32, N, M, C, GESVD, EPS_F32);
    svd_test!(test_f32_gesdd_f_mn, f32, M, N, F, GESDD, EPS_F32);
    svd_test!(test_f32_gesdd_f_nm, f32, N, M, F, GESDD, EPS_F32);
    svd_test!(test_f32_gesdd_c_mn, f32, M, N, C, GESDD, EPS_F32);
    svd_test!(test_f32_gesdd_c_nm, f32, N, M, C, GESDD, EPS_F32);
}
