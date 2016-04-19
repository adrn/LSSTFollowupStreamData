from __future__ import division, print_function

# Standard library
import pickle
import os
import sys

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
import scipy.optimize as so
import emcee
from config import potentials, galcen_frame

# Custom
from gary.util import atleast_2d, get_pool
import gary.coordinates as gc
from gary.dynamics import orbitfit
from gary.dynamics.orbit import combine
import gary.integrate as gi
import gary.potential as gp
from gary.units import galactic

def integrate_forward_backward(potential, w0, t_forw, t_back, dt=0.5,
                               Integrator=gi.DOPRI853Integrator, t0=0.):
    """
    Integrate an orbit forward and backward from a point and combine
    into a single orbit object.

    Parameters
    ----------
    potential : :class:`gary.potential.PotentialBase`
    w0 : :class:`gary.dynamics.CartesianPhaseSpacePosition`, array_like
    t_forw : numeric
        The amount of time to integate forward in time (a positive number).
    t_back : numeric
        The amount of time to integate backwards in time (a negative number).
    dt : numeric (optional)
        The timestep.
    Integrator : :class:`gary.integrate.Integrator` (optional)
        The integrator class to use.
    t0 : numeric (optional)
        The initial time.

    Returns
    -------
    orbit : :class:`gary.dynamics.CartesianOrbit`
    """
    if t_back != 0:
        o1 = potential.integrate_orbit(w0, dt=-dt, t1=t0, t2=t_back, Integrator=Integrator)
    else:
        o1 = None

    if t_forw != 0:
        o2 = potential.integrate_orbit(w0, dt=dt, t1=t0, t2=t_forw, Integrator=Integrator)
    else:
        o2 = None

    if o1 is None:
        return o2
    elif o2 is None:
        return o1

    o1 = o1[::-1]
    o2 = o2[1:]
    orbit = combine((o1, o2), along_time_axis=True)

    if orbit.pos.shape[-1] == 1:
        return orbit[:,0]
    else:
        return orbit

def _unpack(p, freeze=None):
    """ Unpack a parameter vector """

    if freeze is None:
        freeze = dict()

    # these are for the initial conditions
    phi2,d,mul,mub,vr = p[:5]
    count_ix = 5

    # time to integrate forward and backward
    if 't_forw' not in freeze:
        t_forw = p[count_ix]
        count_ix += 1
    else:
        t_forw = freeze['t_forw']

    if 't_back' not in freeze:
        t_back = p[count_ix]
        count_ix += 1
    else:
        t_back = freeze['t_back']

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        phi2_sigma = p[count_ix]
        count_ix += 1
    else:
        phi2_sigma = freeze['phi2_sigma']

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        d_sigma = p[count_ix]
        count_ix += 1
    else:
        d_sigma = freeze['d_sigma']

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        vr_sigma = p[count_ix]
        count_ix += 1
    else:
        vr_sigma = freeze['vr_sigma']

    # ------------------------------------------------------------------------
    # POTENTIAL PARAMETERS
    potential_params = dict()

    for param_name in ['v_c', 'r_h', 'q1', 'q2', 'q3', 'phi']:
        if 'potential_{}'.format(param_name) not in freeze:
            potential_params[param_name] = p[count_ix]
            count_ix += 1
        else:
            potential_params[param_name] = freeze['potential_{}'.format(param_name)]

    return (phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma,potential_params)

def ln_prior(p, data, err, R, Potential, dt, freeze=None):
    """
    Evaluate the prior over stream orbit fit parameters.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    """

    # log prior value
    lp = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma,potential_params = _unpack(p, freeze)

    # time to integrate forward and backward
    t_integ = np.abs(t_forw) + np.abs(t_back)
    if t_forw <= t_back:
        raise ValueError("Forward integration time less than or equal to "
                         "backwards integration time.")

    if (t_forw != 0 and t_forw < dt) or (t_back != 0 and t_back > -dt):
        return -np.inf

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        if phi2_sigma <= 0.:
            return -np.inf
        lp += -np.log(phi2_sigma)

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        if d_sigma <= 0.:
            return -np.inf
        lp += -np.log(d_sigma)

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        if vr_sigma <= 0.:
            return -np.inf
        lp += -np.log(vr_sigma)

    # strong prior on phi2
    if phi2 < -np.pi/2. or phi2 > np.pi/2:
        return -np.inf
    lp += norm.logpdf(phi2, loc=0., scale=phi2_sigma)

    # uniform prior on integration time
    ntimes = int(t_integ / dt) + 1
    if t_integ <= 2. or t_integ > 1000. or ntimes < 4:
        return -np.inf

    if potential_params['v_c'] < 0.1 or potential_params['v_c'] > 0.25:
        return -np.inf

    if potential_params['r_h'] < 1. or potential_params['r_h'] > 50:
        return -np.inf

    if potential_params['q1'] < 0.5 or potential_params['q1'] > 1.:
        return -np.inf

    if potential_params['q2'] < 0.5 or potential_params['q2'] > 1.:
        return -np.inf

    if potential_params['q3'] < 0.5 or potential_params['q3'] > 1.:
        return -np.inf

    if potential_params['phi'] < -np.pi or potential_params['phi'] > np.pi:
        return -np.inf

    return lp

def _mcmc_sample_to_coord(p, R):
    p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
    rep = coord.SphericalRepresentation(lon=p[0]*0.*u.radian,
                                        lat=p[0]*u.radian, # this index looks weird but is right
                                        distance=p[1]*u.kpc)
    return coord.Galactic(orbitfit.rotate_sph_coordinate(rep, R.T))

def _mcmc_sample_to_w0(p, R):
    p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
    c = _mcmc_sample_to_coord(p, R)
    x0 = c.transform_to(galcen_frame['galactocentric_frame']).cartesian.xyz.decompose(galactic).value
    v0 = gc.vhel_to_gal(c, pm=(p[2]*u.rad/u.Myr,p[3]*u.rad/u.Myr), rv=p[4]*u.kpc/u.Myr,
                        **galcen_frame).decompose(galactic).value
    w0 = np.concatenate((x0, v0))
    return w0

def ln_likelihood(p, data, err, R, Potential, dt, freeze=None):
    """ Evaluate the stream orbit fit likelihood. """
    chi2 = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma,potential_params = _unpack(p, freeze)

    w0 = _mcmc_sample_to_w0([phi2,d,mul,mub,vr], R)[:,0]

    # HACK: a prior on velocities
    vmag2 = np.sum(w0[3:]**2)
    chi2 += -vmag2 / (0.15**2)

    # integrate the orbit
    potential = Potential(units=galactic, **potential_params)
    orbit = integrate_forward_backward(potential, w0, t_back=t_back, t_forw=t_forw)

    # rotate the model points to stream coordinates
    model_c,model_v = orbit.to_frame(coord.Galactic, **galcen_frame)
    model_oph = orbitfit.rotate_sph_coordinate(model_c.spherical, R)
#      = model_c.transform_to(Ophiuchus)

    # model stream points in ophiuchus coordinates
    model_phi1 = model_oph.lon
    model_phi2 = model_oph.lat.radian
    model_d = model_oph.distance.decompose(galactic).value
    model_mul,model_mub,model_vr = [x.decompose(galactic).value for x in model_v]

    # for independent variable, use cos(phi)
    data_x = np.cos(data['phi1'])
    model_x = np.cos(model_phi1)
    # data_x = ophdata.coord_oph.phi1.wrap_at(180*u.deg).radian
    # model_x = model_phi1.wrap_at(180*u.deg).radian
    ix = np.argsort(model_x)

    # shortening for readability -- the data
    phi2 = data['phi2'].radian
    dist = data['distance'].decompose(galactic).value
    mul = data['mul'].decompose(galactic).value
    mub = data['mub'].decompose(galactic).value
    vr = data['vr'].decompose(galactic).value

    # define interpolating functions
    order = 3
    # bbox = [-np.pi, np.pi]
    bbox = [-1, 1]
    phi2_interp = InterpolatedUnivariateSpline(model_x[ix], model_phi2[ix], k=order, bbox=bbox) # change bbox to units of model_x
    d_interp = InterpolatedUnivariateSpline(model_x[ix], model_d[ix], k=order, bbox=bbox)
    mul_interp = InterpolatedUnivariateSpline(model_x[ix], model_mul[ix], k=order, bbox=bbox)
    mub_interp = InterpolatedUnivariateSpline(model_x[ix], model_mub[ix], k=order, bbox=bbox)
    vr_interp = InterpolatedUnivariateSpline(model_x[ix], model_vr[ix], k=order, bbox=bbox)

    chi2 += -(phi2_interp(data_x) - phi2)**2 / phi2_sigma**2 - 2*np.log(phi2_sigma)

    _err = err['distance'].decompose(galactic).value
    chi2 += -(d_interp(data_x) - dist)**2 / (_err**2 + d_sigma**2) - np.log(_err**2 + d_sigma**2)

    _err = err['mul'].decompose(galactic).value
    chi2 += -(mul_interp(data_x) - mul)**2 / (_err**2) - 2*np.log(_err)

    _err = err['mub'].decompose(galactic).value
    chi2 += -(mub_interp(data_x) - mub)**2 / (_err**2) - 2*np.log(_err)

    _err = err['vr'].decompose(galactic).value
    chi2 += -(vr_interp(data_x) - vr)**2 / (_err**2 + vr_sigma**2) - np.log(_err**2 + vr_sigma**2)

    # this is some kind of whack prior - don't integrate more than we have to
#     chi2 += -(model_phi1.radian.min() - data['phi1'].radian.min())**2 / (phi2_sigma**2)
#     chi2 += -(model_phi1.radian.max() - data['phi1'].radian.max())**2 / (phi2_sigma**2)

    return 0.5*chi2

def ln_posterior(p, *args, **kwargs):
    """
    Evaluate the stream orbit fit posterior probability.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    Returns
    -------
    lp : float
        The log of the posterior probability.

    """

    lp = ln_prior(p, *args, **kwargs)
    if not np.isfinite(lp):
        return -np.inf

    ll = ln_likelihood(p, *args, **kwargs)
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return lp + ll.sum()

def main(data_file, potential_name, mpi=False, n_walkers=None, n_iterations=None,
         overwrite=False):
    np.random.seed(42)

    pool = get_pool(mpi=mpi)

    # load data file, uncertainties, and potential
    data_file = os.path.abspath(data_file)
    with open(data_file, "rb") as f:
        data,err,R = pickle.load(f) # R is rotation matrix from Galactic to stream coords
    potential = potentials[potential_name]

    freeze = dict()
    # these estimated from the plots
    freeze['phi2_sigma'] = np.radians(0.5)
    freeze['d_sigma'] = 1.5
    freeze['vr_sigma'] = (1*u.km/u.s).decompose(galactic).value
    freeze['t_forw'] = 0.
    if potential_name == 'spherical':
        freeze['t_back'] = -700. # HACK: figured out at bottom of notebook
        potential_freeze_params = ['r_h', 'q1', 'q2', 'q3', 'phi']

    elif potential_name == 'triaxial':
        freeze['t_back'] = -600 # HACK: figured out at bottom of notebook
        potential_freeze_params = ['r_h', 'q1', 'phi']

    for k in potential_freeze_params:
        logger.debug("freezing potential:{}".format(k))
        freeze['potential_{}'.format(k)] = potential.parameters[k].value

    pot_guess = []
    for k in potential.parameters.keys():
        if k in potential_freeze_params: continue
        logger.debug("varying potential:{}".format(k))
        pot_guess += [potential.parameters[k].value]

    idx = data['phi1'].argmin()
    p0_guess = [data['phi2'].radian[idx],
                data['distance'].decompose(galactic).value[idx],
                data['mul'].decompose(galactic).value[idx],
                data['mub'].decompose(galactic).value[idx],
                data['vr'].decompose(galactic).value[idx]]
    p0_guess = p0_guess + pot_guess
    logger.debug("Initial guess: {}".format(p0_guess))

    # first, optimize to get a good guess to initialize MCMC
    args = (data, err, R, potential.__class__, 0.5, freeze)
    logger.info("optimizing ln_posterior...")
    res = so.minimize(lambda *args,**kwargs: -ln_posterior(*args, **kwargs),
                      x0=p0_guess, method='powell', args=args)
    logger.info("finished optimizing")
    logger.debug("optimization returned: {}".format(res))
    if not res.success:
        pool.close()
        raise ValueError("Failed to optimize!")

    # now, create initial conditions for MCMC walkers in a small ball around the
    #   optimized parameter vector
    if n_walkers is None:
        n_walkers = 8*len(p0_guess)
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=len(p0_guess), lnpostfn=ln_posterior,
                                    args=args, pool=pool)
    mcmc_p0 = emcee.utils.sample_ball(res.x, 1E-3*np.array(p0_guess), size=n_walkers)

    if n_iterations is None:
        n_iterations = 1024

    logger.info("running mcmc sampler with {} walkers for {} steps".format(n_walkers, n_iterations))
    _ = sampler.run_mcmc(mcmc_p0, N=n_iterations)
    logger.info("finished sampling")

    pool.close()

    # same sampler to pickle file
    data_file_basename = os.path.splitext(os.path.basename(data_file))[0]
    sampler_path = os.path.join("results", "{}_sampler.pickle".format(data_file_basename))
    if not os.path.exists("results"):
        os.mkdir("results")

    if os.path.exists(sampler_path) and overwrite:
        os.remove(sampler_path)

    sampler.lnprobfn = None
    sampler.pool = None
    logger.debug("saving emcee sampler to: {}".format(sampler_path))
    with open(sampler_path, 'wb') as f:
        pickle.dump(sampler, f)

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. FILES.")

    parser.add_argument("--potential", dest="potential_name", required=True,
                        help="Name of the potential YAML file.")
    parser.add_argument("--data-file", dest="data_file", required=True,
                        help="data file to use.")

    # emcee
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--mcmc-walkers", dest="mcmc_walkers", type=int, default=None,
                        help="Number of walkers.")
    parser.add_argument("--mcmc-steps", dest="mcmc_steps", type=int, required=True,
                        help="Number of steps to take MCMC.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(potential_name=args.potential_name, data_file=args.data_file,
         mpi=args.mpi, n_walkers=args.mcmc_walkers, n_iterations=args.mcmc_steps,
         overwrite=args.overwrite)
