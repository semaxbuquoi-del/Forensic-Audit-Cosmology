import numpy as np
import pandas as pd
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import ConvexHull

def run_manifold_audit():
    """
    Performs a 3D volumetric audit of 20 cosmic structures to correlate 
    Metric Shear Stress with Metric Stiffness (Kurtosis).
    """
    
    # [Jitter (km/s), Kurtosis, RA, Dec, Redshift]
    census_data = {
        "Boötes North":     [2213.47, -1.68, 227.79, 46.0, 0.05],
        "Eridanus V":       [2195.74, -1.52, 53.0, -20.0, 0.04],
        "Perseus-Pisces V": [1935.26, -0.78, 15.0, 36.0, 0.02],
        "Pegasus V":        [1825.86, -1.45, 350.0, 15.0, 0.02],
        "Boötes South":     [1628.84, -0.95, 215.0, 15.0, 0.04],
        "Corona Bor V":     [1627.99, -0.53, 230.0, 30.0, 0.07],
        "Leo Void":         [1514.70, -0.22, 160.0, 15.0, 0.03],
        "Pegasus North V":  [1381.51, 1.98, 345.0, 30.0, 0.04],
        "Ursa Major V":     [1329.20, -0.52, 180.0, 50.0, 0.04],
        "Hercules V":       [1319.16, 1.09, 245.0, 5.0, 0.03],
        "Lynx-Ursa Fil":    [1767.86, -1.19, 135.0, 55.0, 0.03],
        "Virgo-Coma Bridge":[1764.04, -0.17, 185.0, 25.0, 0.02],
        "Leo Supercluster": [1608.14, -1.29, 175.0, 20.0, 0.03],
        "Corona Bor W":     [1531.84, 0.12, 230.0, 35.0, 0.07],
        "Hercules Wall":    [1499.65, -0.28, 240.0, 35.0, 0.03],
        "Coma Supercluster":[1498.69, -0.99, 195.0, 28.0, 0.02],
        "Draco Filament":   [1494.71, -1.04, 250.0, 60.0, 0.03],
        "Sloan Great Wall": [1441.57, -0.63, 202.0, 1.0, 0.07],
        "Cancer Filament":  [1280.11, -0.49, 130.0, 20.0, 0.03],
        "Ursa Major Fil":   [1238.62, 0.10, 170.0, 45.0, 0.04]
    }

    results = []
    print("Initializing SDSS Manifold Audit...")

    for name, stats in census_data.items():
        jitter, kurtosis, ra, dec, z = stats
        
        # Query SDSS for galaxy distribution
        query = f"""
        SELECT ra, dec, z FROM SpecObj 
        WHERE ra BETWEEN {ra-3} AND {ra+3} 
        AND dec BETWEEN {dec-3} AND {dec+3}
        AND z BETWEEN {z-0.03} AND {z+0.03}
        AND class = 'GALAXY' AND zWarning = 0
        """
        
        try:
            data = SDSS.query_sql(query)
            if data is None or len(data) < 4:
                continue
            
            # Convert to Comoving Cartesian Space
            coords = SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg, 
                              distance=cosmo.comoving_distance(data['z']))
            pts = coords.cartesian.xyz.value.T
            
            # Volume & Density Calculation
            hull = ConvexHull(pts)
            density = len(data) / hull.volume
            
            # Physics: Metric Shear Stress (0.5 * rho * v^2)
            shear_stress = 0.5 * density * (jitter**2)
            
            results.append({
                "Structure": name,
                "Density": density,
                "Shear_Stress": shear_stress,
                "Kurtosis": kurtosis
            })
            print(f"Audit Successful: {name}")
        except:
            print(f"Audit Failed: {name} (Coverage Gap)")

    df = pd.DataFrame(results)
    
    # Calculate Final Pearson Correlation
    correlation = df['Shear_Stress'].corr(df['Kurtosis'])
    
    print("\n" + "="*45)
    print("MANIFOLD AUDIT COMPLETE")
    print(f"Final Pearson R (Stress vs Stiffness): {correlation:.4f}")
    print("="*45)
    return df

if __name__ == "__main__":
    df_final = run_manifold_audit()
    print(df_final[['Structure', 'Shear_Stress', 'Kurtosis']])
