# SatQuery AI — 15 Satellite Image Test Dataset

> **Dataset Overview**: Benchmark collection of 15 real high-resolution remote-sensing satellite test images covering multi-class land cover features (water bodies, urban structures, residential housing, roads, vehicles, bridges, boats, vegetation, and agricultural fields).

## Test Cases & Metadata Index

### T1 — Water Body + Houses + Roads + Cars
- **Filename**: `T1.jpg`
- **Source**: LEVIR-CD High-Resolution Remote Sensing Benchmark
- **Geographic Location**: 37°47'24.0"N 122°24'00.0"W (San Francisco Bay Area, USA)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel (High Resolution Aerial/Satellite)
- **Visibly Present Objects**: water_body, houses, roads, cars, vegetation
- **Suggested VQA Questions**:
  - *Is there a water body in this image?*
  - *Are there houses or residential structures visible?*
  - *Is a road network present?*
  - *Are cars or vehicles visible on the roads?*
  - *Is vegetation present near the structures?*
  - *What is the dominant land-cover type?*
  - *Where is the water body located relative to the houses?*

---

### T2 — River + Buildings + Bridge + Vehicles
- **Filename**: `T2.jpg`
- **Source**: LEVIR-CD Urban Remote Sensing Benchmark
- **Geographic Location**: 51°30'18.0"N 0°04'30.0"W (River Thames Corridor, London, UK)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, buildings, bridge, cars, roads
- **Suggested VQA Questions**:
  - *Is a river or water channel visible in this image?*
  - *Are there large commercial or industrial buildings present?*
  - *Is there a bridge spanning the water?*
  - *Are vehicles visible on the bridge or adjacent streets?*
  - *Are roads visible along the embankment?*
  - *What is the dominant land-cover type?*
  - *How many major building structures can be identified?*

---

### T3 — Lake + Residential Buildings + Roads + Vegetation
- **Filename**: `T3.jpg`
- **Source**: High-Res Suburban Lakeshore Benchmark
- **Geographic Location**: 39°05'30.0"N 120°02'30.0"W (Lake Tahoe Shoreline, Nevada, USA)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, houses, roads, vegetation
- **Suggested VQA Questions**:
  - *Is there a lake or large inland water body present?*
  - *Are residential houses visible near the shore?*
  - *Are paved access roads visible connecting the houses?*
  - *Is tree canopy vegetation present in the scene?*
  - *Are boats or piers visible on the lake?*
  - *What is the dominant land-cover type?*
  - *Where is the vegetation located relative to the residential area?*

---

### T4 — Dense Urban Area + Buildings + Cars + Roads
- **Filename**: `T4.jpg`
- **Source**: LEVIR-CD Dense Urban Benchmark
- **Geographic Location**: 48°51'24.0"N 2°21'07.0"E (Central Paris Metro District, France)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: buildings, cars, roads, houses
- **Suggested VQA Questions**:
  - *Is this scene a dense urban area?*
  - *Are high-density building blocks present?*
  - *Is an asphalt road grid clearly visible?*
  - *Are parked or moving cars visible on the streets?*
  - *Is there any natural water body visible?*
  - *What is the dominant land-cover type?*
  - *Are there trees or green spaces within the urban grid?*

---

### T5 — Coastal Area + Buildings + Water + Boats
- **Filename**: `T5.jpg`
- **Source**: Coastal Marine Remote Sensing Dataset
- **Geographic Location**: 45°26'15.0"N 12°20'09.0"E (Venetian Lagoon Coast, Venice, Italy)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, buildings, boats, roads
- **Suggested VQA Questions**:
  - *Is a sea, ocean, or coastal water body present?*
  - *Are shoreline buildings or dock structures visible?*
  - *Are boats or watercraft visible in the water?*
  - *Are access roads or walkways present near the waterfront?*
  - *Are agricultural fields present in this image?*
  - *What is the dominant land-cover type?*
  - *Where are the boats positioned relative to the waterfront structures?*

---

### T6 — Rural Area + Houses + Pond/Lake + Vegetation
- **Filename**: `T6.jpg`
- **Source**: LEVIR-CD Rural Settlement Benchmark
- **Geographic Location**: 50°33'00.0"N 8°40'00.0"E (Hessen Agricultural Zone, Germany)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: houses, water_body, vegetation, roads, fields
- **Suggested VQA Questions**:
  - *Is this scene classified as a rural area?*
  - *Are farmhouses or small rural residences visible?*
  - *Is a small pond or freshwater lake present?*
  - *Is dense green tree canopy or vegetation present?*
  - *Are unpaved or narrow roads visible?*
  - *What is the dominant land-cover type?*
  - *Where is the pond located relative to the rural houses?*

---

### T7 — Agricultural Fields + Roads + Houses
- **Filename**: `T7.jpg`
- **Source**: Agricultural Remote Sensing Dataset
- **Geographic Location**: 38°30'00.0"N 98°00'00.0"W (Central Kansas Croplands, USA)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: fields, roads, houses, vegetation
- **Suggested VQA Questions**:
  - *Are agricultural fields or cultivated crop plots present?*
  - *Are rural access roads dividing the fields?*
  - *Are farm houses or storage structures present?*
  - *Is there a large body of water in this scene?*
  - *Are vehicles visible on the agricultural roads?*
  - *What is the dominant land-cover type?*
  - *What geometric pattern do the agricultural fields form?*

---

### T8 — Industrial Area + Factories + Roads + Vehicles
- **Filename**: `T8.jpg`
- **Source**: LEVIR-CD Industrial Park Benchmark
- **Geographic Location**: 51°53'00.0"N 4°15'00.0"E (Port Industrial Zone, Rotterdam, Netherlands)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: buildings, roads, cars, fields
- **Suggested VQA Questions**:
  - *Is this an industrial or manufacturing zone?*
  - *Are large industrial factory warehouses visible?*
  - *Are wide industrial transport roads visible?*
  - *Are heavy trucks or vehicles present near the facilities?*
  - *Are residential houses present in the immediate vicinity?*
  - *What is the dominant land-cover type?*
  - *Where are the vehicles parked relative to the industrial buildings?*

---

### T9 — Residential Neighborhood + Houses + Cars + Streets
- **Filename**: `T9.jpg`
- **Source**: LEVIR-CD Suburban Neighborhood Dataset
- **Geographic Location**: 30°16'00.0"N 97°44'00.0"W (Suburban Travis County, Austin, Texas, USA)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: houses, cars, roads, vegetation
- **Suggested VQA Questions**:
  - *Is this a residential neighborhood?*
  - *Are single-family houses with visible roof tiles present?*
  - *Are suburban streets or cul-de-sacs visible?*
  - *Are personal cars visible in driveways or on streets?*
  - *Is lawn vegetation or yard trees present?*
  - *What is the dominant land-cover type?*
  - *How regularly arranged are the houses along the streets?*

---

### T10 — Urban Area + River + Buildings + Bridge
- **Filename**: `T10.jpg`
- **Source**: Metropolitan Riverbank Benchmark
- **Geographic Location**: 47°29'50.0"N 19°02'30.0"E (Danube River Core, Budapest, Hungary)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, buildings, bridge, roads, cars
- **Suggested VQA Questions**:
  - *Is there a major river bisecting an urban area?*
  - *Are multi-story metropolitan buildings present on both riverbanks?*
  - *Is a bridge spanning across the river visible?*
  - *Are perimeter roads running parallel to the riverbank?*
  - *Are boats or barges operating on the river?*
  - *What is the dominant land-cover type?*
  - *Where is the bridge located relative to the commercial buildings?*

---

### T11 — Marina/Waterfront + Boats + Buildings + Roads
- **Filename**: `T11.jpg`
- **Source**: LEVIR-CD Coastal Marina Benchmark
- **Geographic Location**: 25°46'00.0"N 80°11'00.0"W (Biscayne Bay Waterfront, Miami, Florida, USA)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, boats, buildings, roads
- **Suggested VQA Questions**:
  - *Is a marina or waterfront dock area present?*
  - *Are boats or yachts moored at the docks?*
  - *Are high-rise coastal buildings visible near the water?*
  - *Is a shoreline boulevard or access road visible?*
  - *Are agricultural fields visible in this scene?*
  - *What is the dominant land-cover type?*
  - *How close are the buildings to the boat docks?*

---

### T12 — Mixed Urban/Rural Area + Fields + Houses + Roads
- **Filename**: `T12.jpg`
- **Source**: LEVIR-CD Mixed Land Use Dataset
- **Geographic Location**: 44°50'00.0"N 7°40'00.0"E (Piedmont Fringe, Italy)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: fields, houses, roads, vegetation, buildings
- **Suggested VQA Questions**:
  - *Does this image depict a mixed urban and rural transition zone?*
  - *Are agricultural fields present alongside houses?*
  - *Are rural residences or farm buildings visible?*
  - *Is a connecting road network visible between fields and houses?*
  - *Is a body of water present in the scene?*
  - *What is the dominant land-cover type?*
  - *What proportion of the image is covered by fields versus housing?*

---

### T13 — Large Water Body + Surrounding Buildings + Vegetation
- **Filename**: `T13.jpg`
- **Source**: Limnological & Urban Shoreline Benchmark
- **Geographic Location**: 46°26'00.0"N 6°33'00.0"E (Lake Geneva Shoreline, Switzerland)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, buildings, vegetation, roads
- **Suggested VQA Questions**:
  - *Is a large water body clearly visible in this image?*
  - *Are surrounding buildings situated along the water perimeter?*
  - *Is green vegetation or forest cover visible nearby?*
  - *Are roads visible encircling the shoreline?*
  - *Are industrial factories visible?*
  - *What is the dominant land-cover type?*
  - *Where is the vegetation situated relative to the water body?*

---

### T14 — Commercial/Urban Area + Parking + Cars + Buildings
- **Filename**: `T14.jpg`
- **Source**: Commercial Retail Remote Sensing Dataset
- **Geographic Location**: 32°46'30.0"N 96°48'00.0"W (Commercial Center, Dallas, Texas, USA)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: buildings, cars, roads, houses
- **Suggested VQA Questions**:
  - *Is this a commercial retail or business district?*
  - *Is a large asphalt parking lot visible?*
  - *Are numerous parked cars visible in the parking lot?*
  - *Are commercial store buildings or shopping centers present?*
  - *Are primary arterial roads bordering the commercial center?*
  - *What is the dominant land-cover type?*
  - *How densely populated with vehicles is the parking lot?*

---

### T15 — Complex Mixed Scene (Water + Houses + Buildings + Roads + Cars + Vegetation + Fields + Boats/Bridge)
- **Filename**: `T15.jpg`
- **Source**: LEVIR-CD Multi-Class Complex Scene Benchmark
- **Geographic Location**: 33°51'30.0"S 151°12'30.0"E (Port Jackson Harbor & Fringe, Sydney, Australia)
- **Resolution / Dimensions**: 256x256 pixels | 0.5m / pixel
- **Visibly Present Objects**: water_body, houses, buildings, roads, cars, vegetation, fields, boats, bridge
- **Suggested VQA Questions**:
  - *Is this a complex mixed landscape containing multiple feature classes?*
  - *Is a water body (bay or river) visible?*
  - *Are residential houses and commercial buildings present?*
  - *Are roads and vehicles visible within the scene?*
  - *Is green vegetation or agricultural land visible?*
  - *Is a bridge or boat dock visible along the water edge?*
  - *What is the dominant land-cover type in this complex scene?*
  - *Where is the dense building cluster located relative to the water and fields?*

---

