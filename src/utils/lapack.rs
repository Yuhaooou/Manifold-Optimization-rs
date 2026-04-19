use core::panic;
use std::{ffi::c_char, ptr::null_mut};

use lapack_sys::{c_double_complex as l_c64, c_float_complex as l_c32, *};
use num_complex::{Complex, Complex32 as c32, Complex64 as c64};

macro_rules! same_type {
    ($t1:ty, $t2:ty) => {
        ::std::any::TypeId::of::<$t1>() == ::std::any::TypeId::of::<$t2>()
    };
}

// pub(super) use same_type;

// macro_rules! is_real_type {
//     ($t:ty) => {
//         same_type!($t, f32) || same_type!($t, f64)
//     };
// }

// pub(super) use is_real_type;

// macro_rules! is_complex_type {
//     ($t:ty) => {
//         same_type!($t, ::num_complex::Complex32) || same_type!($t, ::num_complex::Complex64)
//     };
// }

// pub(super) use is_complex_type;

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

pub trait LapackElem: Default + Clone + Copy + 'static {
    type Real: Default + Clone + Copy + 'static;
}

impl LapackElem for f64 {
    type Real = f64;
}

impl LapackElem for f32 {
    type Real = f32;
}

impl LapackElem for c64 {
    type Real = f64;
}

impl LapackElem for c32 {
    type Real = f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixOrder {
    C,
    F,
}

#[inline]
pub fn to_lapack_complex<T: Copy>(c: &Complex<T>) -> __BindgenComplex<T> {
    // unsafe {
    //     *(c as *const Complex<T> as *const __BindgenComplex<T>)
    // }
    __BindgenComplex { re: c.re, im: c.im }
}

pub fn gesvd_r<T: 'static>(
    jobu: LapackChar,
    jobvt: LapackChar,
    m: i32,
    n: i32,
    mat: *mut T,
    lda: i32,
    s: *mut T,
    u: Option<*mut T>,
    ldu: i32,
    vt: Option<*mut T>,
    ldvt: i32,
    work: *mut T,
    lwork: i32,
) -> i32 {
    let mut info = 0;
    unsafe {
        if same_type!(T, f64) {
            dgesvd_(
                jobu.as_ptr(),
                jobvt.as_ptr(),
                &m,
                &n,
                mat as *mut f64,
                &lda,
                s as *mut f64,
                u.unwrap_or(null_mut()) as *mut f64,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut f64,
                &ldvt,
                work as *mut f64,
                &lwork,
                &mut info,
            );
        } else if same_type!(T, f32) {
            sgesvd_(
                jobu.as_ptr(),
                jobvt.as_ptr(),
                &m,
                &n,
                mat as *mut f32,
                &lda,
                s as *mut f32,
                u.unwrap_or(null_mut()) as *mut f32,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut f32,
                &ldvt,
                work as *mut f32,
                &lwork,
                &mut info,
            );
        } else {
            panic!("Unsupported type for gesvd_r: only f32 and f64 are supported");
        }
    }
    info
}

pub fn gesvd_c<T: 'static>(
    jobu: LapackChar,
    jobvt: LapackChar,
    m: i32,
    n: i32,
    mat: *mut Complex<T>,
    lda: i32,
    s: *mut T,
    u: Option<*mut Complex<T>>,
    ldu: i32,
    vt: Option<*mut Complex<T>>,
    ldvt: i32,
    work: *mut Complex<T>,
    rwork: *mut T,
    lwork: i32,
) -> i32 {
    let mut info = 0;
    unsafe {
        if same_type!(T, f64) {
            zgesvd_(
                jobu.as_ptr(),
                jobvt.as_ptr(),
                &m,
                &n,
                mat as *mut l_c64,
                &lda,
                s as *mut f64,
                u.unwrap_or(null_mut()) as *mut l_c64,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut l_c64,
                &ldvt,
                work as *mut l_c64,
                &lwork,
                rwork as *mut f64,
                &mut info,
            );
        } else if same_type!(T, f32) {
            cgesvd_(
                jobu.as_ptr(),
                jobvt.as_ptr(),
                &m,
                &n,
                mat as *mut l_c32,
                &lda,
                s as *mut f32,
                u.unwrap_or(null_mut()) as *mut l_c32,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut l_c32,
                &ldvt,
                work as *mut l_c32,
                &lwork,
                rwork as *mut f32,
                &mut info,
            );
        } else {
            panic!(
                "Unsupported type for gesvd_c: only Complex<f32> and Complex<f64> are supported"
            );
        }
    }
    info
}

pub fn gesdd_r<T: 'static>(
    jobz: LapackChar,
    m: i32,
    n: i32,
    mat: *mut T,
    lda: i32,
    s: *mut T,
    u: Option<*mut T>,
    ldu: i32,
    vt: Option<*mut T>,
    ldvt: i32,
    work: *mut T,
    lwork: i32,
    iwork: *mut i32,
) -> i32 {
    let mut info = 0;
    unsafe {
        if same_type!(T, f64) {
            dgesdd_(
                jobz.as_ptr(),
                &m,
                &n,
                mat as *mut f64,
                &lda,
                s as *mut f64,
                u.unwrap_or(null_mut()) as *mut f64,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut f64,
                &ldvt,
                work as *mut f64,
                &lwork,
                iwork as *mut i32,
                &mut info,
            );
        } else if same_type!(T, f32) {
            sgesdd_(
                jobz.as_ptr(),
                &m,
                &n,
                mat as *mut f32,
                &lda,
                s as *mut f32,
                u.unwrap_or(null_mut()) as *mut f32,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut f32,
                &ldvt,
                work as *mut f32,
                &lwork,
                iwork as *mut i32,
                &mut info,
            );
        } else {
            panic!("Unsupported type for gesdd_r: only f32 and f64 are supported");
        }
    }
    info
}

pub fn gesdd_c<T: 'static>(
    jobz: LapackChar,
    m: i32,
    n: i32,
    mat: *mut Complex<T>,
    lda: i32,
    s: *mut T,
    u: Option<*mut Complex<T>>,
    ldu: i32,
    vt: Option<*mut Complex<T>>,
    ldvt: i32,
    work: *mut Complex<T>,
    lwork: i32,
    rwork: *mut T,
    iwork: *mut i32,
) -> i32 {
    let mut info = 0;
    unsafe {
        if same_type!(T, f64) {
            zgesdd_(
                jobz.as_ptr(),
                &m,
                &n,
                mat as *mut l_c64,
                &lda,
                s as *mut f64,
                u.unwrap_or(null_mut()) as *mut l_c64,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut l_c64,
                &ldvt,
                work as *mut l_c64,
                &lwork,
                rwork as *mut f64,
                iwork as *mut i32,
                &mut info,
            );
        } else if same_type!(T, f32) {
            cgesdd_(
                jobz.as_ptr(),
                &m,
                &n,
                mat as *mut l_c32,
                &lda,
                s as *mut f32,
                u.unwrap_or(null_mut()) as *mut l_c32,
                &ldu,
                vt.unwrap_or(null_mut()) as *mut l_c32,
                &ldvt,
                work as *mut l_c32,
                &lwork,
                rwork as *mut f32,
                iwork as *mut i32,
                &mut info,
            );
        } else {
            panic!(
                "Unsupported type for gesdd_c: only Complex<f32> and Complex<f64> are supported"
            );
        }
    }
    info
}
