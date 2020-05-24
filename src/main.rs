use clap::Clap;
use fdm::base::{Equation, Simluation};
use fdm::equations::{Advection, InviscidBurger};
use fdm::schemes::{BeamWarming, LaxFriedrichs, LaxWendroff, Scheme, Upwind};
use gnuplot::{AxesCommon, Figure, Fix, Font};
use itertools::iproduct;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;

#[derive(Clap)]
struct Args {
    #[clap(short, long, default_value = "outputs")]
    output_dir: String,
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir).unwrap();

    // equations
    let mut eqs: HashMap<String, Box<dyn Equation<f64>>> = HashMap::new();
    eqs.insert("Advection".into(), Box::new(Advection { a: 1.0 }));
    eqs.insert("InviscidBurger".into(), Box::new(InviscidBurger));

    // conditions
    let dx = 1e-2;
    let cfl = 0.6;
    let dt = cfl * dx;
    let boundary = [-3., 3.];

    // schemes
    let mut schemes: HashMap<String, Box<dyn Scheme<f64>>> = HashMap::new();
    schemes.insert("Upwind".into(), Box::new(Upwind));
    schemes.insert("BeamWarming".into(), Box::new(BeamWarming));
    schemes.insert("LaxWendroff".into(), Box::new(LaxWendroff));
    schemes.insert("LaxFriedrichs".into(), Box::new(LaxFriedrichs));

    // initial waves
    let mut inits: HashMap<String, Box<dyn Fn(f64) -> f64>> = HashMap::new();
    inits.insert("Sine".into(), Box::new(|x: f64| (PI * x).sin()));
    inits.insert(
        "Square".into(),
        Box::new(|x: f64| if x >= 0. && x <= 1. { 1.0 } else { 0. }),
    );

    for ((eq_name, eq), (init_name, init), (scheme_name, scheme)) in
        iproduct!(eqs.iter(), inits.iter(), schemes.iter())
    {
        let mut fig = Figure::new();
        let name = format!("{}-{}-{}", eq_name, init_name, scheme_name);
        println!("Processing {}", name);
        fig.set_title(&name).set_terminal(
            "gif animate optimize delay 2 size 480,360",
            &format!("{}/{}.gif", args.output_dir, name),
        );

        let mut sim = Simluation::<f64>::new(dx, dt, boundary, init);

        for i in 0..(3. / dt) as i32 {
            if i > 0 {
                fig.new_page();
            }
            let ax = fig
                .axes2d()
                .set_title(&name, &[Font("Times", 20.0)])
                .set_x_grid(true)
                .set_y_grid(true)
                .set_y_range(Fix(-1.5), Fix(1.5))
                .set_x_range(Fix(boundary[0]), Fix(boundary[1]));

            sim.set_state(scheme.run(&sim, &**eq));
            ax.lines(&sim.grid, &sim.state, &[]);
        }

        fig.show().unwrap();
    }
}
