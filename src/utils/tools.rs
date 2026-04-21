use core::fmt;
use std::error::Error;

use ndarray::{ScalarOperand, prelude::*};

use crate::utils::traits::RCLike;

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
    D: RCLike + ScalarOperand,
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
    D: RCLike + ScalarOperand,
{
    assert!(
        mat.shape()[0] == mat.shape()[1],
        "Input must be a square matrix."
    );
    (mat - &mat.t()) / D::from_i8(2).unwrap()
}