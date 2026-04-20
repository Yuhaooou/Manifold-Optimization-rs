use std::{ffi::c_char, fmt::Debug, ptr::null_mut};

use lapack_sys::*;
use num_complex::{Complex, Complex32 as c32, Complex64 as c64, ComplexFloat};
use num_traits::Zero;

use crate::utils::lapack;

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

pub trait LapackElem: ComplexFloat + 'static + Debug {
    const IS_REAL: bool;

    fn from_real(r: Self::Real) -> Self;

    /// For real types, return 0. For complex types, get the imaginary unit.
    fn get_i() -> Self;
}

impl LapackElem for f64 {
    const IS_REAL: bool = true;

    fn from_real(r: Self::Real) -> Self {
        r
    }

    fn get_i() -> Self {
        Self::zero()
    }
}

impl LapackElem for f32 {
    const IS_REAL: bool = true;

    fn from_real(r: Self::Real) -> Self {
        r
    }

    fn get_i() -> Self {
        Self::zero()
    }
}

impl LapackElem for c64 {
    const IS_REAL: bool = false;

    fn from_real(r: Self::Real) -> Self {
        Complex::new(r, 0.)
    }

    fn get_i() -> Self {
        Complex::new(0., 1.)
    }
}

impl LapackElem for c32 {
    const IS_REAL: bool = false;

    fn from_real(r: Self::Real) -> Self {
        Complex::new(r, 0.)
    }

    fn get_i() -> Self {
        Complex::new(0., 1.)
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

pub trait LapackGESVD: LapackElem {
    fn gesvd(
        jobu: LapackChar,
        jobvt: LapackChar,
        m: i32,
        n: i32,
        mat: *mut Self,
        lda: i32,
        s: *mut Self::Real,
        u: Option<*mut Self>,
        ldu: i32,
        vt: Option<*mut Self>,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
    ) -> (i32, Option<Vec<Self::Real>>);
}

macro_rules! lapack_gesvd_r {
    ($t:ty, $fun:expr) => {
        impl LapackGESVD for $t {
            fn gesvd(
                jobu: LapackChar,
                jobvt: LapackChar,
                m: i32,
                n: i32,
                mat: *mut Self,
                lda: i32,
                s: *mut Self::Real,
                u: Option<*mut Self>,
                ldu: i32,
                vt: Option<*mut Self>,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
            ) -> (i32, Option<Vec<Self::Real>>) {
                let mut info = 0;
                unsafe {
                    $fun(
                        jobu.as_ptr(),
                        jobvt.as_ptr(),
                        &m,
                        &n,
                        mat as *mut $t,
                        &lda,
                        s as *mut $t,
                        u.unwrap_or(null_mut()) as *mut $t,
                        &ldu,
                        vt.unwrap_or(null_mut()) as *mut $t,
                        &ldvt,
                        work as *mut $t,
                        &lwork,
                        &mut info,
                    );
                    (info, None)
                }
            }
        }
    };
}

lapack_gesvd_r!(f64, lapack::dgesvd_);
lapack_gesvd_r!(f32, lapack::sgesvd_);

macro_rules! lapack_gesvd_c {
    ($t:ty, $fun:expr) => {
        impl LapackGESVD for Complex<$t> {
            fn gesvd(
                jobu: LapackChar,
                jobvt: LapackChar,
                m: i32,
                n: i32,
                mat: *mut Self,
                lda: i32,
                s: *mut Self::Real,
                u: Option<*mut Self>,
                ldu: i32,
                vt: Option<*mut Self>,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
            ) -> (i32, Option<Vec<Self::Real>>) {
                let mut info = 0;
                let mut rwork = new_uninit_vec(5 * m.min(n) as usize);
                unsafe {
                    $fun(
                        jobu.as_ptr(),
                        jobvt.as_ptr(),
                        &m,
                        &n,
                        mat as *mut __BindgenComplex<$t>,
                        &lda,
                        s as *mut $t,
                        u.unwrap_or(null_mut()) as *mut __BindgenComplex<$t>,
                        &ldu,
                        vt.unwrap_or(null_mut()) as *mut __BindgenComplex<$t>,
                        &ldvt,
                        work as *mut __BindgenComplex<$t>,
                        &lwork,
                        rwork.as_mut_ptr(),
                        &mut info,
                    );
                    (info, Some(rwork))
                }
            }
        }
    };
}

lapack_gesvd_c!(f64, lapack::zgesvd_);
lapack_gesvd_c!(f32, lapack::cgesvd_);

pub trait LapackGESDD: LapackElem {
    fn gesdd(
        jobz: LapackChar,
        m: i32,
        n: i32,
        mat: *mut Self,
        lda: i32,
        s: *mut Self::Real,
        u: Option<*mut Self>,
        ldu: i32,
        vt: Option<*mut Self>,
        ldvt: i32,
        work: *mut Self,
        rwork: Option<*mut Self::Real>,
        lwork: i32,
    ) -> (i32, Vec<i32>);
}

macro_rules! lapack_gesdd_r {
    ($t:ty, $fun:expr) => {
        impl LapackGESDD for $t {
            fn gesdd(
                jobz: LapackChar,
                m: i32,
                n: i32,
                mat: *mut Self,
                lda: i32,
                s: *mut Self::Real,
                u: Option<*mut Self>,
                ldu: i32,
                vt: Option<*mut Self>,
                ldvt: i32,
                work: *mut Self,
                _rwork: Option<*mut Self::Real>,
                lwork: i32,
            ) -> (i32, Vec<i32>) {
                let mut info = 0;
                let mut iwork = new_uninit_vec(8 * m.min(n) as usize);
                unsafe {
                    $fun(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        mat as *mut $t,
                        &lda,
                        s as *mut $t,
                        u.unwrap_or(null_mut()) as *mut $t,
                        &ldu,
                        vt.unwrap_or(null_mut()) as *mut $t,
                        &ldvt,
                        work as *mut $t,
                        &lwork,
                        iwork.as_mut_ptr(),
                        &mut info,
                    );
                    (info, iwork)
                }
            }
        }
    };
}

lapack_gesdd_r!(f64, lapack::dgesdd_);
lapack_gesdd_r!(f32, lapack::sgesdd_);

macro_rules! lapack_gesdd_c {
    ($t:ty, $fun:expr) => {
        impl LapackGESDD for Complex<$t> {
            fn gesdd(
                jobz: LapackChar,
                m: i32,
                n: i32,
                mat: *mut Self,
                lda: i32,
                s: *mut Self::Real,
                u: Option<*mut Self>,
                ldu: i32,
                vt: Option<*mut Self>,
                ldvt: i32,
                work: *mut Self,
                rwork: Option<*mut Self::Real>,
                lwork: i32,
            ) -> (i32, Vec<i32>) {
                let mut info = 0;
                let mut iwork = new_uninit_vec(8 * m.min(n) as usize);
                unsafe {
                    $fun(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        mat as *mut __BindgenComplex<$t>,
                        &lda,
                        s as *mut $t,
                        u.unwrap_or(null_mut()) as *mut __BindgenComplex<$t>,
                        &ldu,
                        vt.unwrap_or(null_mut()) as *mut __BindgenComplex<$t>,
                        &ldvt,
                        work as *mut __BindgenComplex<$t>,
                        &lwork,
                        rwork.unwrap_or(null_mut()) as *mut $t,
                        iwork.as_mut_ptr(),
                        &mut info,
                    );
                    (info, iwork)
                }
            }
        }
    };
}

lapack_gesdd_c!(f64, lapack::zgesdd_);
lapack_gesdd_c!(f32, lapack::cgesdd_);

pub trait LapackRoutines: LapackGESVD + LapackGESDD {}

impl<T> LapackRoutines for T where T: LapackGESVD + LapackGESDD {}
