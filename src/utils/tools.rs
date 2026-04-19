use core::fmt;
use std::error::Error;

use ndarray::{ScalarOperand, prelude::*};
use ndarray_linalg::{Lapack, QR, SVD};
use num_traits::Float;

use crate::utils::traits::Real;

/// Utility errors used by numerical helpers.
#[derive(Debug)]
pub enum Errors {
    ShapeError(String),
    ComputeError(String),
}

impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Errors::ShapeError(info) => write!(f, "Shape error! {}", info),
            Errors::ComputeError(name) => write!(f, "Compute {} Error!", name),
        }
    }
}

impl Error for Errors {}

/// Return the symmetric part of a square matrix: `(A + A^T) / 2`.
///
/// # Panics
/// Panics if `mat` is not square.
pub fn mat_sym<D>(mat: &Array2<D>) -> Array2<D>
where
    D: Real + ScalarOperand,
{
    assert!(
        mat.shape()[0] == mat.shape()[1],
        "Input must be a square matrix."
    );
    (mat + &mat.t()) / D::from_i8(2).unwrap()
}

/// Return the skew-symmetric part of a square matrix: `(A - A^T) / 2`.
///
/// # Panics
/// Panics if `mat` is not square.
pub fn mat_skew<D>(mat: &Array2<D>) -> Array2<D>
where
    D: Real + ScalarOperand,
{
    assert!(
        mat.shape()[0] == mat.shape()[1],
        "Input must be a square matrix."
    );
    (mat - &mat.t()) / D::from_i8(2).unwrap()
}

/// QR decomposition helper.
///
/// Returns the `Q` factor adjusted by the sign of `diag(R)`.
pub fn qr<D>(mat: &Array2<D>) -> Result<Array2<D>, Errors>
where
    D: Lapack,
    D::Real: Float,
{
    if let Ok((q, r)) = mat.qr() {
        let sign = r.diag().mapv(|x| D::from_real(x.re().signum()));
        Ok(q * sign)
    } else {
        Err(Errors::ComputeError("QR Decomposition".to_string()))
    }
}

/// Truncated SVD helper.
///
/// Because ndarray-linalg does not expose a dedicated truncated SVD API,
/// this function computes full SVD and slices the leading `r` components.
pub fn tsvd<D>(mat: &Array2<D>) -> Result<(Array2<D>, Array1<D>, Array2<D>), Errors>
where
    D: Lapack<Real = D>,
{
    let r = mat.shape()[0].min(mat.shape()[1]);
    // TODO: directly use *gesvd to get truncated SVD.
    if let Ok((Some(u), s, Some(vt))) = mat.svd(true, true) {
        let u_truncated = u.slice(s![.., ..r]).to_owned();
        let s_truncated = s.slice(s![..r]).to_owned();
        let vt_truncated = vt.slice(s![..r, ..]).to_owned();
        Ok((u_truncated, s_truncated, vt_truncated))
    } else {
        Err(Errors::ComputeError("Truncated SVD".to_string()))
    }
}
