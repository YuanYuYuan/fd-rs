pub mod base;
pub mod equations;
pub mod schemes;

pub use base::Equation;
pub use schemes::Scheme;

pub type BoxedEquation = Box<dyn Equation<f64> + Send + Sync + 'static>;
pub type BoxedScheme = Box<dyn Scheme<f64> + Sync + Send + 'static>;
pub type BoxedFunction = Box<dyn Fn(f64) -> f64 + Sync + Send + 'static>;
