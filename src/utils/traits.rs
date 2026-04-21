use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_complex::{Complex32 as c32, Complex64 as c64, ComplexFloat};
use num_traits::{Float, FromPrimitive, One, Zero};

pub trait FieldOps:
    Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
}

impl<T> FieldOps for T where
    T: Sized
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
{
}

pub trait Field: Zero + One + FieldOps + Clone {}

impl<T> Field for T where T: Zero + One + FieldOps + Clone {}

pub trait RCLike: Field + ImaginaryUnit + ComplexFloat + FromPrimitive + 'static + IsReal {
    fn from_real(real: Self::Real) -> Self {
        Self::from(real).unwrap()
    }

    /// For real types, return self. For complex types, return self multiplied by i.
    fn try_from_real_to_imag(real: Self::Real) -> Self {
        Self::from_real(real).mul_i_or_one()
    }
}

impl<T> RCLike for T where T: Field + ImaginaryUnit + ComplexFloat + FromPrimitive + 'static + IsReal
{}

pub trait Real: RCLike<Real = Self> + Float {
    /// Return 0.5.
    fn half() -> Self {
        Self::from(0.5).unwrap()
    }

    fn addi(self, rhs: i32) -> Self {
        self + Self::from_i32(rhs).unwrap()
    }
    fn subi(self, rhs: i32) -> Self {
        self - Self::from_i32(rhs).unwrap()
    }
    fn muli(self, rhs: i32) -> Self {
        self * Self::from_i32(rhs).unwrap()
    }
    fn divi(self, rhs: i32) -> Self {
        self / Self::from_i32(rhs).unwrap()
    }

    fn sqrt_(self) -> Self {
        Float::sqrt(self)
    }

    fn abs_(self) -> Self {
        Float::abs(self)
    }

    fn powi_(self, exp: i32) -> Self {
        Float::powi(self, exp)
    }

    fn powf_(self, exp: Self) -> Self {
        Float::powf(self, exp)
    }
}

impl<T> Real for T where T: RCLike<Real = Self> + Float {}

pub trait ImaginaryUnit {
    /// Return 0 for real types and i for complex types.
    fn get_i_or_zero() -> Self;

    /// Multiply self by i if it is complex, otherwise return self.
    fn mul_i_or_one(self) -> Self;
}

impl ImaginaryUnit for c64 {
    fn get_i_or_zero() -> Self {
        c64::new(0.0, 1.0)
    }

    fn mul_i_or_one(self) -> Self {
        self * c64::new(0.0, 1.0)
    }
}

impl ImaginaryUnit for c32 {
    fn get_i_or_zero() -> Self {
        c32::new(0.0, 1.0)
    }

    fn mul_i_or_one(self) -> Self {
        self * c32::new(0.0, 1.0)
    }
}

impl ImaginaryUnit for f64 {
    fn get_i_or_zero() -> Self {
        0.0
    }

    fn mul_i_or_one(self) -> Self {
        self
    }
}

impl ImaginaryUnit for f32 {
    fn get_i_or_zero() -> Self {
        0.0
    }

    fn mul_i_or_one(self) -> Self {
        self
    }
}

pub trait IsReal {
    const IS_REAL: bool;
}

impl IsReal for c64 {
    const IS_REAL: bool = false;
}

impl IsReal for c32 {
    const IS_REAL: bool = false;
}

impl IsReal for f64 {
    const IS_REAL: bool = true;
}

impl IsReal for f32 {
    const IS_REAL: bool = true;
}

/// Algebraic vector operations used by manifold tangent vectors.
pub trait Vector:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + for<'c> Add<&'c Self, Output = Self>
    + for<'c> Sub<&'c Self, Output = Self>
    + Mul<Self::Field, Output = Self>
    + Div<Self::Field, Output = Self>
    + Neg<Output = Self>
{
    type Field: RCLike;

    /// Sum all elements.
    fn sum(&self) -> Self::Field;

    // Definition of vector space operations. Here we expose right multiplication
    // because it is more convenient in Rust code.
    fn ref_mul_num(&self, num: Self::Field) -> Self;

    // For computational convenience.
    fn ref_div_num(&self, num: Self::Field) -> Self;

    fn ref_add(&self, rhs: Self) -> Self;

    fn ref_sub(&self, rhs: Self) -> Self;

    fn ref_add_ref(&self, rhs: &Self) -> Self;

    fn ref_sub_ref(&self, rhs: &Self) -> Self;

    fn ref_add_num(&self, num: Self::Field) -> Self;

    fn ref_sub_num(&self, num: Self::Field) -> Self;

    fn elementwise_mul(&self, rhs: &Self) -> Self;

    fn elementwise_div(&self, rhs: &Self) -> Self;

    // TODO: Find a way to get shape.
    // As the shape of the ndarray is not a type, zeros(), ones() and from_elem() cannot be implemented.
    // Instead, we choose to get it from a shape or a known ndarray.
    fn zeros() -> Self {
        panic!("Zeros is not implemented for this vector type.")
    }

    fn ones() -> Self {
        panic!("Ones is not implemented for this vector type.")
    }

    fn from_elem(elem: Self::Field) -> Self {
        let _ = elem;
        panic!("from_elem is not implemented for this vector type.")
    }

    fn zeros_with_shape<Shape>(shape: &Shape) -> Self {
        let _ = shape;
        panic!("zeros_with_shape is not implemented for this vector type.")
    }

    fn ones_with_shape<Shape>(shape: &Shape) -> Self {
        let _ = shape;
        panic!("ones_with_shape is not implemented for this vector type.")
    }

    fn nums_with_shape<Shape>(num: Self::Field, shape: &Shape) -> Self {
        let _ = (num, shape);
        panic!("from_elem_with_shape is not implemented for this vector type.")
    }

    fn zeros_like(&self) -> Self;

    fn ones_like(&self) -> Self;

    fn nums_like(&self, num: Self::Field) -> Self;
}

pub trait Norm {
    type Field: RCLike;

    /// 2-norm for vectors, F-norm for matrices and tensors.
    fn norm(&self) -> Self::Field;
}

/// Convenience alias: a vector with norm support.
pub trait NormedVector<K>: Vector<Field = K> + Norm<Field = K> {}

/// Inner product trait for vector-like objects.
pub trait InnerProduct {
    type Field: RCLike;

    /// Inner product with `rhs`.
    fn inner(&self, rhs: &Self) -> <Self::Field as ComplexFloat>::Real;
}

/// Convenience alias: a normed vector with inner product.
pub trait InnerProductVector<K>: NormedVector<K> + InnerProduct<Field = K> {}
