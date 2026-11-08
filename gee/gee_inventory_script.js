/**************************************************************************************
 * Kodagu 2018 landslide candidate inventory — Sentinel-2 NDVI change
 * Manuscript AGJSR-02-2026-0031 (revision 0711). Author: Vinay Kellengere Shankarnarayan
 *
 * HOW TO RUN
 *   1. Paste this whole script into https://code.earthengine.google.com (RVIM account).
 *   2. Click "Run". Check the Console: it prints the number of candidate patches.
 *   3. Open the "Tasks" tab (top right) and click "RUN" next to
 *      Kodagu_landslide_candidates_v2, then "Run" in the dialog.
 *   4. Click "Save" and name the script kodagu_inventory_2018.
 *
 * METHOD (reported in Section 2.2 of the manuscript)
 *   Imagery   : Sentinel-2 Level-1C top-of-atmosphere reflectance (COPERNICUS/S2_HARMONIZED).
 *               Level-1C is used because Level-2A surface reflectance is not available
 *               for this region before December 2018.
 *   Clouds    : Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED), cs_cdf >= 0.60.
 *   Composites: median NDVI, same season one year apart (removes phenology):
 *               before = 1 Oct 2017 – 28 Feb 2018; after = 1 Oct 2018 – 28 Feb 2019.
 *   Candidate : NDVI_before >= 0.50 (vegetated) AND NDVI_before − NDVI_after >= 0.20
 *               AND NDVI_after < 0.35 (bare or sparse after the 2018 monsoon)
 *               AND not water (MNDWI_after < 0).
 *   Patches   : 8-connected groups of >= 9 pixels (>= 900 m² at 10 m), vectorised.
 *   Output    : centroid longitude/latitude and patch area (m²) for every patch.
 *               Screening (district boundary, area <= 2 ha, centroid slope >= 15°) is done
 *               afterwards in Python (02_samples.py), so it is fully scripted and logged.
 **************************************************************************************/

// Study region: rectangle enclosing Kodagu district (the district boundary is applied in Python)
var region = ee.Geometry.Rectangle([75.30, 11.90, 76.25, 12.92]);

var S2  = ee.ImageCollection('COPERNICUS/S2_HARMONIZED');
var CSP = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var CLEAR = 0.60;

function composite(start, end) {
  var col = S2.filterBounds(region).filterDate(start, end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
              .linkCollection(CSP, ['cs_cdf'])
              .map(function (img) {
                var ndvi  = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
                var mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI');
                return ndvi.addBands(mndwi).updateMask(img.select('cs_cdf').gte(CLEAR));
              });
  print('Scenes ' + start + ' to ' + end + ':', col.size());
  return col.median().clip(region);
}

var before = composite('2017-10-01', '2018-03-01');
var after  = composite('2018-10-01', '2019-03-01');

var dNDVI = before.select('NDVI').subtract(after.select('NDVI')).rename('dNDVI');
var cand = before.select('NDVI').gte(0.50)
             .and(dNDVI.gte(0.20))
             .and(after.select('NDVI').lt(0.35))
             .and(after.select('MNDWI').lt(0))
             .selfMask();

// keep patches of at least 9 connected 10 m pixels (900 m2)
var size = cand.connectedPixelCount(256, true);
cand = cand.updateMask(size.gte(9));

var patches = cand.reduceToVectors({
  geometry: region, scale: 10, geometryType: 'polygon', eightConnected: true,
  labelProperty: 'cand', maxPixels: 1e10, bestEffort: false
});
patches = patches.map(function (f) {
  var c = f.geometry().centroid(1);
  var xy = c.coordinates();
  return ee.Feature(c, {
    longitude: xy.get(0), latitude: xy.get(1),
    area_m2: f.geometry().area(1)
  });
});

print('Candidate patches (>= 900 m2):', patches.size());
Map.centerObject(region, 10);
Map.addLayer(dNDVI, {min: -0.2, max: 0.6, palette: ['white', 'orange', 'red']}, 'dNDVI 2017/18 → 2018/19', false);
Map.addLayer(cand, {palette: ['red']}, 'Candidate scars');

Export.table.toDrive({
  collection: patches.select(['longitude', 'latitude', 'area_m2']),
  description: 'Kodagu_landslide_candidates_v2',
  folder: 'GEE_Kodagu_v2',
  fileNamePrefix: 'Kodagu_landslide_candidates_v2',
  fileFormat: 'CSV',
  selectors: ['longitude', 'latitude', 'area_m2']
});
