# Based on similar code form mescaline by Hayley J. Macpherson

# A script to generate theta,phi coordinates
# and translate to Cartesian to be read-in to mescaline
#
import healpy as hp
import numpy as np

# Choose N_side and output filename
Nside = 1 # Number of light rays = 12*Nside*Nside

# ---------------------------------

# Number of pixels from Nside
npix       = hp.nside2npix(Nside)

# Set up indices from 0,N_pix
indices    = np.arange(0,npix,1)

# Generate spherical coords for these pixels and translate to Cartesian x,y,z
theta, phi = hp.pix2ang(Nside,indices)
xyzs       = hp.ang2vec(theta,phi)



np.save(f"data/healpix_{Nside}.npy", xyzs)