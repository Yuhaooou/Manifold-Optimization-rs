use std::{
    fmt::{Debug, Display, LowerExp, UpperExp},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ndarray_linalg::Scalar;
use num_complex::{Complex32 as c32, Complex64 as c64};
use num_traits::{Float as NumFloat, Num, NumCast, One, Zero};

// Use it always cause multiple applicable items in scope error, so we will not use it.
// use num_traits::real::Real as NumReal;

/// Real-number trait bound used throughout optimization code.
pub trait Real:
    Num
    + Copy
    + NumCast
    + PartialOrd
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Debug
    + Display
    + NumFloat
    + LowerExp
    + UpperExp
    + 'static
{
    /// Convert from `f64`.
    fn from_f64(f: f64) -> Self;

    /// Convert to `f64`.
    fn to_f64(self) -> f64;

    /// Convert from `f32`.
    fn from_f32(f: f32) -> Self;

    /// Convert to `f32`.
    fn to_f32(self) -> f32;

    /// Return 0.5.
    fn half() -> Self {
        Self::from_f64(0.5)
    }

    fn fromi8(i: i8) -> Self {
        Self::from_f32(i as f32)
    }

    fn addi(self, rhs: i32) -> Self;
    fn subi(self, rhs: i32) -> Self;
    fn muli(self, rhs: i32) -> Self;
    fn divi(self, rhs: i32) -> Self;

    // fn max(self, other: Self) -> Self {
    //     NumFloat::max(self, other)
    // }

    // fn min(self, other: Self) -> Self {
    //     NumFloat::min(self, other)
    // }

    // fn sqrt(self) -> Self {
    //     NumFloat::sqrt(self)
    // }

    // fn cos(self) -> Self {
    //     NumFloat::cos(self)
    // }

    // fn sin(self) -> Self {
    //     NumFloat::sin(self)
    // }

    // fn acos(self) -> Self {
    //     NumFloat::acos(self)
    // }

    // fn asin(self) -> Self {
    //     NumFloat::asin(self)
    // }
}

impl Real for f64 {
    fn from_f64(f: f64) -> Self {
        f
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f32(f: f32) -> Self {
        f as f64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn addi(self, rhs: i32) -> Self {
        self + (rhs as f64)
    }

    fn subi(self, rhs: i32) -> Self {
        self - (rhs as f64)
    }

    fn muli(self, rhs: i32) -> Self {
        self * (rhs as f64)
    }

    fn divi(self, rhs: i32) -> Self {
        self / (rhs as f64)
    }
}

impl Real for f32 {
    fn from_f64(f: f64) -> Self {
        f as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f32(f: f32) -> Self {
        f
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn addi(self, rhs: i32) -> Self {
        self + (rhs as f32)
    }

    fn subi(self, rhs: i32) -> Self {
        self - (rhs as f32)
    }

    fn muli(self, rhs: i32) -> Self {
        self * (rhs as f32)
    }

    fn divi(self, rhs: i32) -> Self {
        self / (rhs as f32)
    }
}

/// Scalar field abstraction used by vectors and manifolds.
pub trait Field:
    Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Debug
    + Clone
{
}

impl Field for c64 {}

impl Field for c32 {}

impl<R> Field for R where R: Real {}

/// Real-or-complex scalar abstraction used by generic vectors.
pub trait RCLike: Field + Copy + 'static {
    // Not like [`Scalar`], where [`Scalar::abs`] returns [`Scalar::Real`] type, here it just
    // returns [`Self`] type. The return type here theoretically ture, because the modulus of
    // a complex number is a real number, which is still a complex number. Besides, the return type
    // of [`Scalar::abs`] is not theoretically equal to [`Self::Real`] type, the theoretical return
    // type should be somethings like $\mathbb{R} \ge 0$, which is hard to implement in Rust.
    /// Absolute value (modulus) of the Real (complex) number. Returns the same type of the input.
    fn abs(self) -> Self;

    fn conj(self) -> Self;

    /// Principal square root.
    fn sqrt(self) -> Self;
}

impl RCLike for c64 {
    fn abs(self) -> Self {
        c64::norm(self).as_c()
    }

    fn conj(self) -> Self {
        c64::conj(&self)
    }

    fn sqrt(self) -> Self {
        c64::sqrt(self)
    }
}

impl RCLike for c32 {
    fn abs(self) -> Self {
        c32::norm(self).as_c()
    }

    fn conj(self) -> Self {
        c32::conj(&self)
    }

    fn sqrt(self) -> Self {
        c32::sqrt(self)
    }
}

impl<R> RCLike for R
where
    R: Real,
{
    fn abs(self) -> R {
        <R as NumFloat>::abs(self)
    }

    fn conj(self) -> Self {
        self
    }

    fn sqrt(self) -> Self {
        <R as NumFloat>::sqrt(self)
    }
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
    + Debug
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

// The tangent space of a (Riemannian) manifold at a point is indeed a normed and inner-product space,
// but these properties are gained from the manifold. So these traits are not required for the
// Manifold::TangentVector. Maybe these traits have no use.
pub trait Norm {
    /// Underlying real scalar type for the norm.
    type Real: Real;

    /// Norm value.
    fn norm(&self) -> Self::Real;
}

/// Convenience alias: a vector with norm support.
pub trait NormedVector<R, K>: Vector<Field = K> + Norm<Real = R> {}

/// Inner product trait for vector-like objects.
pub trait InnerProduct {
    /// Underlying real scalar type for the inner product.
    type Real: Real;

    /// Inner product with `rhs`.
    fn inner(&self, rhs: &Self) -> Self::Real;
}

/// Convenience alias: a normed vector with inner product.
pub trait InnerProductVector<R, K>: NormedVector<R, K> + InnerProduct<Real = R> {}
