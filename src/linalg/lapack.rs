use std::{ffi::c_char, ptr::null_mut};

use lapack_sys::*;
use num_complex::{Complex, Complex32 as c32, Complex64 as c64};
use openblas_src as _; // Ensure OpenBLAS is linked

use crate::utils::traits::RCLike;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapackChar {
    A,
    S,
    O,
    C,
    N,
    L,
    R,
    V,
    I,
}

impl LapackChar {
    fn as_ptr(self) -> *const c_char {
        match self {
            LapackChar::A => &(b'A' as c_char),
            LapackChar::S => &(b'S' as c_char),
            LapackChar::O => &(b'O' as c_char),
            LapackChar::C => &(b'C' as c_char),
            LapackChar::N => &(b'N' as c_char),
            LapackChar::L => &(b'L' as c_char),
            LapackChar::R => &(b'R' as c_char),
            LapackChar::V => &(b'V' as c_char),
            LapackChar::I => &(b'I' as c_char),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "a" => LapackChar::A,
            "s" => LapackChar::S,
            "o" => LapackChar::O,
            "c" => LapackChar::C,
            "n" => LapackChar::N,
            "l" => LapackChar::L,
            "r" => LapackChar::R,
            "v" => LapackChar::V,
            "i" => LapackChar::I,
            _ => unimplemented!("Unsupported Lapack character: {}", s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    C,
    F,
}

impl Layout {
    pub fn is_c(self) -> bool {
        self == Layout::C
    }

    pub fn is_f(self) -> bool {
        self == Layout::F
    }
}

#[inline]
pub fn to_lapack_complex<T: Copy>(c: &Complex<T>) -> __BindgenComplex<T> {
    // unsafe {
    //     *(c as *const Complex<T> as *const __BindgenComplex<T>)
    // }
    __BindgenComplex { re: c.re, im: c.im }
}

pub(crate) fn new_uninit_vec<T>(len: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(len);
    unsafe { vec.set_len(len) };
    vec
}

fn as_mut_ptr_or_null<T>(opt: Option<&mut [T]>) -> *mut T {
    match opt {
        Some(slice) => slice.as_mut_ptr(),
        None => null_mut(),
    }
}

pub(crate) trait LapackGESVD: RCLike {
    fn gesvd(
        jobu: LapackChar,
        jobvt: LapackChar,
        m: i32,
        n: i32,
        mat: &mut [Self],
        lda: i32,
        s: &mut [Self::Real],
        u: Option<&mut [Self]>,
        ldu: i32,
        vt: Option<&mut [Self]>,
        ldvt: i32,
    ) -> (i32, Vec<Self::Real>);
}

macro_rules! lapack_gesvd_r {
    ($t:ty, $fun:expr) => {
        impl LapackGESVD for $t {
            fn gesvd(
                jobu: LapackChar,
                jobvt: LapackChar,
                m: i32,
                n: i32,
                mat: &mut [Self],
                lda: i32,
                s: &mut [Self::Real],
                u: Option<&mut [Self]>,
                ldu: i32,
                vt: Option<&mut [Self]>,
                ldvt: i32,
            ) -> (i32, Vec<Self::Real>) {
                let mut info = 0;
                let mut work = new_uninit_vec(1);
                let u_ptr = as_mut_ptr_or_null(u);
                let vt_ptr = as_mut_ptr_or_null(vt);
                unsafe {
                    $fun(
                        jobu.as_ptr(),
                        jobvt.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr(),
                        &lda,
                        s.as_mut_ptr(),
                        u_ptr,
                        &ldu,
                        vt_ptr,
                        &ldvt,
                        work.as_mut_ptr(),
                        &(-1),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in gesvd workspace query: {}", info);
                }
                let lwork = work[0] as usize;
                let mut work = new_uninit_vec(lwork);
                unsafe {
                    $fun(
                        jobu.as_ptr(),
                        jobvt.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr(),
                        &lda,
                        s.as_mut_ptr(),
                        u_ptr,
                        &ldu,
                        vt_ptr,
                        &ldvt,
                        work.as_mut_ptr(),
                        &(lwork as i32),
                        &mut info,
                    );
                }
                if info < 0 {
                    panic!("Illegal value in gesvd argument: {}", -info);
                }
                (info, work)
            }
        }
    };
}

lapack_gesvd_r!(f64, dgesvd_);
lapack_gesvd_r!(f32, sgesvd_);

macro_rules! lapack_gesvd_c {
    ($t:ty, $fun:expr) => {
        impl LapackGESVD for Complex<$t> {
            fn gesvd(
                jobu: LapackChar,
                jobvt: LapackChar,
                m: i32,
                n: i32,
                mat: &mut [Self],
                lda: i32,
                s: &mut [Self::Real],
                u: Option<&mut [Self]>,
                ldu: i32,
                vt: Option<&mut [Self]>,
                ldvt: i32,
            ) -> (i32, Vec<Self::Real>) {
                let mut info = 0;
                let mut rwork = new_uninit_vec(5 * m.min(n) as usize);
                let mut work = new_uninit_vec(1);
                let u_ptr = as_mut_ptr_or_null(u);
                let vt_ptr = as_mut_ptr_or_null(vt);
                unsafe {
                    $fun(
                        jobu.as_ptr(),
                        jobvt.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &lda,
                        s.as_mut_ptr() as *mut $t,
                        u_ptr as *mut __BindgenComplex<$t>,
                        &ldu,
                        vt_ptr as *mut __BindgenComplex<$t>,
                        &ldvt,
                        work.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &(-1),
                        rwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in gesvd workspace query: {}", info);
                }
                let lwork = (work[0] as __BindgenComplex<$t>).re as usize;
                let mut work = new_uninit_vec(lwork);
                unsafe {
                    $fun(
                        jobu.as_ptr(),
                        jobvt.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &lda,
                        s.as_mut_ptr() as *mut $t,
                        u_ptr as *mut __BindgenComplex<$t>,
                        &ldu,
                        vt_ptr as *mut __BindgenComplex<$t>,
                        &ldvt,
                        work.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &(lwork as i32),
                        rwork.as_mut_ptr(),
                        &mut info,
                    )
                }
                if info < 0 {
                    panic!("Illegal value in gesvd argument: {}", -info);
                }
                (info, rwork)
            }
        }
    };
}

lapack_gesvd_c!(f64, zgesvd_);
lapack_gesvd_c!(f32, cgesvd_);

pub(crate) trait LapackGESDD: RCLike {
    fn gesdd(
        jobz: LapackChar,
        m: i32,
        n: i32,
        mat: &mut [Self],
        lda: i32,
        s: &mut [Self::Real],
        u: Option<&mut [Self]>,
        ldu: i32,
        vt: Option<&mut [Self]>,
        ldvt: i32,
        lrwork: i32,
    ) -> i32;
}

macro_rules! lapack_gesdd_r {
    ($t:ty, $fun:expr) => {
        impl LapackGESDD for $t {
            fn gesdd(
                jobz: LapackChar,
                m: i32,
                n: i32,
                mat: &mut [Self],
                lda: i32,
                s: &mut [Self::Real],
                u: Option<&mut [Self]>,
                ldu: i32,
                vt: Option<&mut [Self]>,
                ldvt: i32,
                _lrwork: i32,
            ) -> i32 {
                let mut info = 0;
                let mut iwork = new_uninit_vec(8 * m.min(n) as usize);
                let mut work = new_uninit_vec(1);
                let u_ptr = as_mut_ptr_or_null(u);
                let vt_ptr = as_mut_ptr_or_null(vt);
                unsafe {
                    $fun(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr(),
                        &lda,
                        s.as_mut_ptr(),
                        u_ptr,
                        &ldu,
                        vt_ptr,
                        &ldvt,
                        work.as_mut_ptr(),
                        &(-1),
                        iwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in gesdd workspace query: {}", info);
                }
                let lwork = work[0] as usize;
                let mut work = new_uninit_vec(lwork);
                unsafe {
                    $fun(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr(),
                        &lda,
                        s.as_mut_ptr(),
                        u_ptr,
                        &ldu,
                        vt_ptr,
                        &ldvt,
                        work.as_mut_ptr(),
                        &(lwork as i32),
                        iwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                if info < 0 {
                    panic!("Illegal value in gesdd argument: {}", -info);
                }
                info
            }
        }
    };
}

lapack_gesdd_r!(f64, dgesdd_);
lapack_gesdd_r!(f32, sgesdd_);

macro_rules! lapack_gesdd_c {
    ($t:ty, $fun:expr) => {
        impl LapackGESDD for Complex<$t> {
            fn gesdd(
                jobz: LapackChar,
                m: i32,
                n: i32,
                mat: &mut [Self],
                lda: i32,
                s: &mut [Self::Real],
                u: Option<&mut [Self]>,
                ldu: i32,
                vt: Option<&mut [Self]>,
                ldvt: i32,
                lrwork: i32,
            ) -> i32 {
                let mut info = 0;
                let mut iwork = new_uninit_vec(8 * m.min(n) as usize);
                let mut work = new_uninit_vec(1);
                let u_ptr = as_mut_ptr_or_null(u);
                let vt_ptr = as_mut_ptr_or_null(vt);
                unsafe {
                    $fun(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &lda,
                        s.as_mut_ptr() as *mut $t,
                        u_ptr as *mut __BindgenComplex<$t>,
                        &ldu,
                        vt_ptr as *mut __BindgenComplex<$t>,
                        &ldvt,
                        work.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &(-1),
                        null_mut() as *mut $t,
                        iwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in gesdd workspace query: {}", info);
                }
                let lwork = (work[0] as __BindgenComplex<$t>).re as usize;
                let mut work = new_uninit_vec(lwork);
                let mut rwork = new_uninit_vec(lrwork as usize);
                unsafe {
                    $fun(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        mat.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &lda,
                        s.as_mut_ptr() as *mut $t,
                        u_ptr as *mut __BindgenComplex<$t>,
                        &ldu,
                        vt_ptr as *mut __BindgenComplex<$t>,
                        &ldvt,
                        work.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &(lwork as i32),
                        rwork.as_mut_ptr(),
                        iwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                if info < 0 {
                    panic!("Illegal value in gesdd argument: {}", -info);
                }
                info
            }
        }
    };
}

lapack_gesdd_c!(f64, zgesdd_);
lapack_gesdd_c!(f32, cgesdd_);

pub trait LapackGEQR: RCLike {
    // fn geqr(
    //     m: i32,
    //     n: i32,
    //     mat: &mut [Self],
    //     lda: i32,
    //     t: &mut [Self],
    //     tsize: i32,
    //     work: &mut [Self],
    //     lwork: i32,
    // ) -> i32;
}

macro_rules! lapack_geqr {
    ($t:ty, $tt:ty, $fun:expr) => {
        impl LapackGEQR for $t {
            // fn geqr(
            //     m: i32,
            //     n: i32,
            //     mat: &mut [Self],
            //     lda: i32,
            //     t: &mut [Self],
            //     tsize: i32,
            //     work: &mut [Self],
            //     lwork: i32,
            // ) -> i32 {
            //     unimplemented!("GEQR");
            // }
        }
    };
}

lapack_geqr!(f64, f64, dgeqr_);
lapack_geqr!(f32, f32, sgeqr_);
lapack_geqr!(c64, __BindgenComplex<f64>, zgeqr_);
lapack_geqr!(c32, __BindgenComplex<f32>, cgeqr_);

pub trait LapackGEQRF: RCLike {
    fn geqrf(m: i32, n: i32, mat: &mut [Self], lda: i32, t: &mut [Self]) -> i32;
}

pub trait LapackGEQRFP: RCLike {
    fn geqrfp(m: i32, n: i32, mat: &mut [Self], lda: i32, t: &mut [Self]) -> i32;
}

fn from_rclike_to_usize<T: RCLike>(x: T) -> usize {
    x.to_usize().unwrap()
}

macro_rules! lapack_geqrf {
    ($trait:ident, $trait_fun:ident, $t:ty, $tt:ty, $ffi_fun:expr) => {
        impl $trait for $t {
            fn $trait_fun(m: i32, n: i32, mat: &mut [Self], lda: i32, t: &mut [Self]) -> i32 {
                let mut info = 0;
                let mut work = new_uninit_vec(1);
                unsafe {
                    $ffi_fun(
                        &m,
                        &n,
                        mat.as_mut_ptr() as *mut $tt,
                        &lda,
                        t.as_mut_ptr() as *mut $tt,
                        work.as_mut_ptr() as *mut $tt,
                        &(-1),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in geqrf workspace query: {}", info);
                }
                let work0 = unsafe { *(&work[0] as *const $tt as *const $t) };
                let lwork = from_rclike_to_usize(work0);
                let mut work = new_uninit_vec(lwork);
                unsafe {
                    $ffi_fun(
                        &m,
                        &n,
                        mat.as_mut_ptr() as *mut $tt,
                        &lda,
                        t.as_mut_ptr() as *mut $tt,
                        work.as_mut_ptr() as *mut $tt,
                        &(lwork as i32),
                        &mut info,
                    );
                }
                if info < 0 {
                    panic!("Illegal value in geqrf argument: {}", -info);
                }
                info
            }
        }
    };
}

lapack_geqrf!(LapackGEQRF, geqrf, f64, f64, dgeqrf_);
lapack_geqrf!(LapackGEQRF, geqrf, f32, f32, sgeqrf_);
lapack_geqrf!(LapackGEQRF, geqrf, c64, __BindgenComplex<f64>, zgeqrf_);
lapack_geqrf!(LapackGEQRF, geqrf, c32, __BindgenComplex<f32>, cgeqrf_);

// Same for geqrfp.
lapack_geqrf!(LapackGEQRFP, geqrfp, f64, f64, dgeqrfp_);
lapack_geqrf!(LapackGEQRFP, geqrfp, f32, f32, sgeqrfp_);
lapack_geqrf!(LapackGEQRFP, geqrfp, c64, __BindgenComplex<f64>, zgeqrfp_);
lapack_geqrf!(LapackGEQRFP, geqrfp, c32, __BindgenComplex<f32>, cgeqrfp_);

pub trait LapackQfrom: RCLike {
    fn qfrom(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &mut [Self]) -> i32;
}

macro_rules! Qfrom_r {
    ($t:ty, $fun:expr) => {
        impl LapackQfrom for $t {
            fn qfrom(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &mut [Self]) -> i32 {
                let mut info = 0;
                let mut work = new_uninit_vec(1);
                unsafe {
                    $fun(
                        &m,
                        &n,
                        &k,
                        a.as_mut_ptr() as *mut $t,
                        &lda,
                        tau.as_mut_ptr() as *mut $t,
                        work.as_mut_ptr() as *mut $t,
                        &(-1),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in qfrom workspace query: {}", info);
                }
                let lwork = work[0] as usize;
                let mut work = new_uninit_vec(lwork);
                unsafe {
                    $fun(
                        &m,
                        &n,
                        &k,
                        a.as_mut_ptr() as *mut $t,
                        &lda,
                        tau.as_mut_ptr() as *mut $t,
                        work.as_mut_ptr() as *mut $t,
                        &(lwork as i32),
                        &mut info,
                    );
                }
                if info < 0 {
                    panic!("Illegal value in qfrom argument: {}", -info);
                }
                info
            }
        }
    };
}

Qfrom_r!(f64, dorgqr_);
Qfrom_r!(f32, sorgqr_);

macro_rules! Qfrom_c {
    ($t:ty, $fun:expr) => {
        impl LapackQfrom for Complex<$t> {
            fn qfrom(m: i32, n: i32, k: i32, a: &mut [Self], lda: i32, tau: &mut [Self]) -> i32 {
                let mut info = 0;
                let mut work = new_uninit_vec(1);
                unsafe {
                    $fun(
                        &m,
                        &n,
                        &k,
                        a.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &lda,
                        tau.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        work.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &(-1),
                        &mut info,
                    );
                }
                if info != 0 {
                    panic!("Error in qfrom workspace query: {}", info);
                }
                let lwork = (unsafe {
                    *(&work[0] as *const __BindgenComplex<$t> as *const Complex<$t>)
                })
                .re as usize;
                let mut work = new_uninit_vec(lwork);
                unsafe {
                    $fun(
                        &m,
                        &n,
                        &k,
                        a.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &lda,
                        tau.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        work.as_mut_ptr() as *mut __BindgenComplex<$t>,
                        &(lwork as i32),
                        &mut info,
                    );
                }
                if info < 0 {
                    panic!("Illegal value in qfrom argument: {}", -info);
                }
                info
            }
        }
    };
}

Qfrom_c!(f64, zungqr_);
Qfrom_c!(f32, cungqr_);

macro_rules! LapackElem {
    ( $( $t:ident ),* ) => {
        #[allow(private_bounds)]
        pub trait LapackElem:
            RCLike
            $(
                + $t
            )*
        {
        }

        impl<T> LapackElem for T where
            T: RCLike
            $(
                + $t
            )*
        {
        }
    };
}

LapackElem!(
    LapackGESVD,
    LapackGESDD,
    LapackGEQR,
    LapackGEQRF,
    LapackGEQRFP,
    LapackQfrom
);
