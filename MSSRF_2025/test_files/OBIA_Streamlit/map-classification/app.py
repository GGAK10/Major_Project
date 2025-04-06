
import streamlit as st
import ee
import geemap
import folium


ee.Initialize(project='bb-kagg09160')

# Define the geometry using an asset from Google Earth Engine
asset_id = 'projects/bb-kagg09160/assets/Velacheri'  # Replace with your actual asset ID
geom = ee.FeatureCollection(asset_id)  # Reference the asset as a FeatureCollection

# Load Sentinel-2 imagery (top-of-atmosphere reflectance, 10m resolution)
image = ee.ImageCollection('COPERNICUS/S2') \
            .filterBounds(geom)  \
            .filterDate('2022-01-01', '2022-12-31')  \
            .median()

# Select relevant bands (e.g., Red, Green, Blue, NIR, SWIR for classification)
bands = image.select(['B4', 'B3', 'B2', 'B8', 'B11'])  # B4 = Red, B3 = Green, B2 = Blue, B8 = NIR, B11 = SWIR

# Perform image segmentation (SLIC algorithm)
segmentation = ee.Algorithms.Image.Segmentation.SNIC(
    image=bands,  # Pass 'bands' as a named argument
    size=30,       # Spatial size of segments (adjust based on image resolution)
    compactness=10 # Balance between segment shape and color similarity
)

# Create sample points for classification (manual training)
points = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([80.212, 13.082]), {'class': 0}),  # Example: Water
  ee.Feature(ee.Geometry.Point([80.227, 13.100]), {'class': 1}),  # Example: Forest
  ee.Feature(ee.Geometry.Point([80.250, 13.080]), {'class': 2}),  # Example: Urban
  ee.Feature(ee.Geometry.Point([80.240, 13.020]), {'class': 3})   # Example: Bare land
])

# Sample the image at the points (extract features for classification)
training = image.sampleRegions(
    collection=points,  # Pass 'points' as the first positional argument
    properties=['class'],
    scale=10
)

# Train a classifier (e.g., Random Forest)
classifier = ee.Classifier.smileRandomForest(50).train(
    training,  # Pass training as the first positional argument
    'class',    # Pass 'class' as the second positional argument (classProperty)
    bands.bandNames()  # Pass bands.bandNames() as the third positional argument (inputProperties)
)

# Classify the image using the trained classifier
classified = image.classify(classifier)

# Create the folium map for Streamlit integration
# Create a geemap Map object
Map = geemap.Map(center=[13.082, 80.227], zoom=12)

# Add the Earth Engine layer to the geemap Map
Map.addLayer(classified.clip(geom), {'min': 0, 'max': 3, 'palette': ['yellow', 'blue', 'red', 'green']}, 'LULC Classification')

# Display the geemap Map in Streamlit
Map.to_streamlit(height=600)

#m = geemap.Map(location=[13.082, 80.227], zoom_start=12)
Map.add_ee_layer(classified.clip(geom), {
    'min': 0, 'max': 3, 'palette': ['yellow', 'blue', 'red', 'green']
}, 'LULC Classification')

# Embed the folium map in the Streamlit app
st.title("Google Earth Engine - LULC Classification")
st.markdown("This is the LULC classification map using Sentinel-2 imagery.")

# Display the folium map using Streamlit
st.write("LULC Classification Map")
Map.to_streamlit(height=600)
#folium_map_html = Map._repr_html_()  # Get the HTML representation of the map

# Display the map in Streamlit using components.html
#st.components.v1.html(folium_map_html, width=800, height=600) 