use std::fmt::Display;

use ndarray::prelude::*;

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
fn thin_svd_r_owned_impl<T: LapackSVD>(
    mut mat: Array2<T>,
    order: MatrixOrder,
    backend: SVDBackend,
) -> (Array2<T>, Array1<T::Real>, Array2<T>) {
    // For f-order mat, directly use lapack routines to compute. For c-order mat, we compute mat.t(), so m and n are reversed.
    let (m, n) = if order.is_f() {
        mat.dim()
    } else {
        (mat.shape()[1], mat.shape()[0])
    };
    let r = m.min(n);
    let mut svd_s = Vec::with_capacity(r);
    let mut svd_u_or_vt = Vec::with_capacity(r * r);
    unsafe {
        svd_s.set_len(r);
        svd_u_or_vt.set_len(r * r);
    }
    let vec_u = if m >= n {
        None
    } else {
        Some(svd_u_or_vt.as_mut_ptr())
    };
    let vec_vt = if m >= n {
        Some(svd_u_or_vt.as_mut_ptr())
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
                svd_s.as_mut_ptr(),
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
            let (info2, _) = T::gesvd(
                jobu,
                jobvt,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                svd_s.as_mut_ptr(),
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
            let mut work = [T::zero()];
            let (info1, _) = T::gesdd(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                svd_s.as_mut_ptr(),
                vec_u,
                m as i32,
                vec_vt,
                n as i32,
                work.as_mut_ptr(),
                None,
                -1_i32,
            );
            if info1 != 0 {
                println!("gesvd_r returned non-zero info: {}", info1);
            }
            let work_len = work[0];
            let mut work = Vec::with_capacity(work_len.to_usize().unwrap());
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
                    let mut vec = Vec::with_capacity(rwork_len);
                    unsafe { vec.set_len(rwork_len) };
                    vec
                }),
            };
            let (info2, _) = T::gesdd(
                LapackChar::O,
                m as i32,
                n as i32,
                mat.as_mut_ptr(),
                m as i32,
                svd_s.as_mut_ptr(),
                vec_u,
                m as i32,
                vec_vt,
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
                println!("gesvd_r returned non-zero info: {}", info2);
            }
        }
        _ => unimplemented!("SVD backend {} is not implemented yet", backend),
    }

    let res_u;
    let res_s;
    let res_vt;
    unsafe {
        res_s = Array::from_shape_vec_unchecked(r, svd_s);
        let u_or_vt = match order {
            MatrixOrder::F => Array::from_shape_vec_unchecked((r, r).f(), svd_u_or_vt),
            MatrixOrder::C => Array::from_shape_vec_unchecked((r, r), svd_u_or_vt),
        };
        if m >= n {
            res_vt = u_or_vt;
            res_u = mat;
        } else {
            res_u = u_or_vt;
            res_vt = mat;
        }
    }

    // For f-order, just return. For c-order, the output of lapack routines are (vt.t(), s, u.t()), need to transpose back.
    match order {
        MatrixOrder::F => (res_u, res_s, res_vt),
        MatrixOrder::C => (res_vt, res_s, res_u),
    }
}

#[allow(private_bounds)]
/// Thin svd for owned matrix. This funtion will destroy the input matrix.
pub fn thin_svd_r_owned<T: LapackSVD>(
    mat: Array2<T>,
    backend: SVDBackend,
) -> (Array2<T>, Array1<T::Real>, Array2<T>) {
    if mat.t().is_standard_layout() {
        thin_svd_r_owned_impl(mat, MatrixOrder::F, backend)
    } else if mat.is_standard_layout() {
        thin_svd_r_owned_impl(mat, MatrixOrder::C, backend)
    } else {
        let _ = mat.view();
        unimplemented!("Input matrix must be either C-order or F-order");
    }
}

mod test {
    #![allow(dead_code, unused)]
    use ndarray_rand::RandomExt;
    use num_complex::{Complex32 as c32, Complex64 as c64, ComplexDistribution};
    use num_traits::Float;
    use rand_distr::{Uniform, uniform::SampleUniform};

    use super::{MatrixOrder::*, SVDBackend::*, *};

    const M: usize = 5;
    const N: usize = 4;
    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-5;

    fn test_svd_inner<T>(mat: Array2<T>, backend: SVDBackend, eps: T::Real)
    where
        T: LapackSVD,
        T::Real: Float,
    {
        let (u, s, vt) = thin_svd_r_owned(mat.clone(), backend);
        let s = s.mapv(T::from_real);
        let a_re = u.dot(&Array2::from_diag(&s)).dot(&vt);
        let err = (a_re - mat)
            .into_iter()
            .map(T::abs)
            .reduce(T::Real::max)
            .unwrap_or(T::zero().re());
        assert!(err < eps);
    }

    fn test_svd<T>(m: usize, n: usize, order: MatrixOrder, backend: SVDBackend, eps: T::Real)
    where
        T: LapackSVD,
        T::Real: SampleUniform,
    {
        let sh = match order {
            F => (m, n).f(),
            C => (m, n).into_shape_with_order(),
        };
        let a = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let b = Array::random(sh, Uniform::new(T::zero().re(), T::one().re()).unwrap());
        let c = a.mapv(T::from_real) + b.mapv(|x| T::from_real(x) * T::get_i());
        test_svd_inner(c, backend, eps);
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

    svd_test!(test_c64_gesvd_f_mn, c64, M, N, F, GESVD, EPS_F64);
    svd_test!(test_c64_gesvd_f_nm, c64, N, M, F, GESVD, EPS_F64);
    svd_test!(test_c64_gesvd_c_mn, c64, M, N, C, GESVD, EPS_F64);
    svd_test!(test_c64_gesvd_c_nm, c64, N, M, C, GESVD, EPS_F64);
    svd_test!(test_c64_gesdd_f_mn, c64, M, N, F, GESDD, EPS_F64);
    svd_test!(test_c64_gesdd_f_nm, c64, N, M, F, GESDD, EPS_F64);
    svd_test!(test_c64_gesdd_c_mn, c64, M, N, C, GESDD, EPS_F64);
    svd_test!(test_c64_gesdd_c_nm, c64, N, M, C, GESDD, EPS_F64);
    svd_test!(test_c32_gesvd_f_mn, c32, M, N, F, GESVD, EPS_F32);
    svd_test!(test_c32_gesvd_f_nm, c32, N, M, F, GESVD, EPS_F32);
    svd_test!(test_c32_gesvd_c_mn, c32, M, N, C, GESVD, EPS_F32);
    svd_test!(test_c32_gesvd_c_nm, c32, N, M, C, GESVD, EPS_F32);
    svd_test!(test_c32_gesdd_f_mn, c32, M, N, F, GESDD, EPS_F32);
    svd_test!(test_c32_gesdd_f_nm, c32, N, M, F, GESDD, EPS_F32);
    svd_test!(test_c32_gesdd_c_mn, c32, M, N, C, GESDD, EPS_F32);
    svd_test!(test_c32_gesdd_c_nm, c32, N, M, C, GESDD, EPS_F32);
}
