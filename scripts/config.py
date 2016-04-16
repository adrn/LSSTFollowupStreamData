from collections import OrderedDict
import astropy.coordinates as coord
import astropy.units as u
import gary.potential as gp
from gary.units import galactic

# Galactocentric reference frame to use for this project
galactocentric_frame = coord.Galactocentric(z_sun=0.*u.pc,
                                            galcen_distance=8.3*u.kpc)
vcirc = 238.*u.km/u.s
vlsr = [-11.1, 12.24, 7.25]*u.km/u.s

galcen_frame = dict()
galcen_frame['galactocentric_frame'] = galactocentric_frame
galcen_frame['vcirc'] = vcirc
galcen_frame['vlsr'] = vlsr

potentials = OrderedDict()
potentials['spherical'] = gp.LogarithmicPotential(v_c=175*u.km/u.s, r_h=20., q1=1., q2=1., q3=1., units=galactic)
potentials['triaxial'] = gp.LogarithmicPotential(v_c=175*u.km/u.s, r_h=20., q1=1., q2=0.95, q3=0.9, units=galactic)
