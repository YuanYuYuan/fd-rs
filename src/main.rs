use gnuplot::{AxesCommon, Figure, Fix};
use std::f64::consts::PI;
use fd_rs::base::Simluation;
use fd_rs::equations::{Advection, InviscidBurger};
use fd_rs::schemes::{Scheme, Upwind, LaxWendroff};

fn main() {
    let eq = Advection::<f64> { a: 1.0 };
    // let eq = InviscidBurger;

    let dx = 1e-2;
    let cfl = 0.6;
    let dt = cfl * dx;

    // let init = |x: f64| (PI * x).sin();
    let init = |x: f64| if x >= 0. && x <= 1. { 1.0 } else { 0. };

    let mut sim = Simluation::<f64>::new(dx, dt, [-5., 5.], init);
    // sim.plot("test");

    // let scheme = Upwind;
    let scheme = LaxWendroff;

    let mut fig = Figure::new();
    fig.set_terminal("gif animate optimize delay 2 size 480,360", "gif.gif");

    for i in 0..(3. / dt) as i32 {
        if i > 0 { fig.new_page(); }
        let ax = fig
            .axes2d()
            .set_x_grid(true)
            .set_y_grid(true)
            .set_y_range(Fix(-2.0), Fix(2.0))
            .set_x_range(Fix(-5.0), Fix(5.0));

        sim.set_state(scheme.run(&sim, &eq));
        ax.lines(&sim.grid, &sim.state, &[]);
    }
    fig.show().unwrap();
}
