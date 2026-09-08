# A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics

[https://doi.org/10.5061/dryad.m37pvmd99](https://doi.org/10.5061/dryad.m37pvmd99)

## General Information

1. Description: This dataset includes measured photovoltaic (PV) power generation data and on-site weather data collected from 60 grid-connected rooftop PV stations in Hong Kong over a three-year period (2021-2023). The PV power generation data was collected at 5-minute intervals. The meteorological data was collected at 1-minute intervals from an on-site weather station. The metadata was represented using Brick schema was developed, which simplifies the data comprehension and the development of smart analytics applications.
2. Date of data collection: 2021-01-01 to 2023-12-31
3. Geographic location of data collection: Sai Kung District, Hong Kong, China(22.3363°N 114.2634°E)
4. Funding sources that supported the collection of the data: The data collection was supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project No. C6003-22Y), and by the National Key R&D Program of China (2023YFC3807100).

## Description of the data and file structure

The open-sourced dataset is divided into two categories: time-series data and metadata. Longitudinal PV generation and meteorological data are provided in .csv format, while metadata of the data measurements is represented by the Brick model in .ttl format.

1. Time-series dataset folder: This folder contains two subfolders, Meteorological dataset and PV generation dataset. The Meteorological dataset subfolder contains seven sub-subfolders, each of which contains three .csv files corresponding to time series data at one-minute resolution for the years 2021, 2022, and 2023. The PV generation dataset subfolder contains two sub-subfolders: PV stations with panel level optimizer and PV stations without panel level optimizer. The former contains two sub-subfolders: Inverter level dataset and Site level dataset, with 44 and 37 .csv files respectively. The latter contains one sub-subfolder: Site level dataset, with 23 .csv files.
2. Metadata folder: PV generation system metadata.ttl is the Brick model of the dataset, which represents the location, equipment, and temporal information for PV systems.
3. The time-series data was pre-processed by replacing missing values with "NA" and resampling the data to ensure temporal consistency.
4. Relationship between files, if important: Each .csv file under the time series dataset is uniquely associated with a specific PV station or inverter name in the Brick model.
5. Additional related data collected that was not included in the current data package: N/A
6. Are there multiple versions of the dataset? No

## Methodological Information

1. Data collection: For stations without panel level optimizers (comprising 23 stations, accounting for 38.3% of the total), the data was individually measured and transferred by the inverter. For stations equipped with panel level optimizers (comprising 37 stations, accounting for 61.7% of the total), the PV generation data was measured and transferred by both the inverter and the panel level optimizer. Meteorological data is collected from the weather station located on the eastern side of the campus. The station comprises a 10-meter-high automatic weather tower and an outdoor plinth area that housing 6 samplers that measure meteorological data at 1-minute intervals.
2. Data transmission and storage: The collected PV generation data was transferred to wireless gateway using dedicated Wi-Fi. Meteorological data is initially sensed, transmitted, and stored in the data logger using RS232 or RS485 communication protocols, and subsequently transferred to the server via LAN cables. All streams of PV generation and meteorological data were extracted and consolidated into a centralized database.
3. Instrument- or software-specific information needed to interpret the data: The time-series data is stored in Comma-Separated Values (CSV) format, which can be further processed and analyzed using various software tools, such as Microsoft Excel or code-based interpreters. The Brick model is available in Turtle (TTL) format, which can be interpreted using the Brick TTL Viewer ([https://viewer.brickschema.org/](https://viewer.brickschema.org/)).
4. Environmental/experimental conditions: The data were collected from 60 grid-connected rooftop PV stations and 1 weather station from a subtropical university campus under real operating conditions.
5. PV system management: The rooftop solar power project is managed by the HKUST Sustainability/Net-Zero Office ([https://sust.hkust.edu.hk](https://sust.hkust.edu.hk)), and was initiated in December 2020.  The PV stations and sensors undergo regular maintenance as per the specifications outlined by the Hong Kong Electrical and Mechanical Services Department (EMSD) to ensure their proper functioning.

## Sharing/Access information

1. Licenses/restrictions placed on the data: N/A
2. Links to publications that cite or use the data: N/A
3. Links to other publicly accessible locations of the data: N/A
4. Links/relationships to ancillary data sets: N/A
5. Was data derived from another source? No
6. Recommended citation for this dataset: Z. Lin, Q. Zhou, Z. Wang, C. Wang, D. Bookhart, M. Leung-Shea. A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics. submitted to Nature Scientific Data, under review.

## Code

An example Python code for querying and retrieving information of PV generation system is available at the dataset’s GitHub page: [https://github.com/ZinanLin-Oscar/SPARQL-Example-for-PV-Brick-Model](https://github.com/ZinanLin-Oscar/SPARQL-Example-for-PV-Brick-Model)

## Version changes

**8-sept-2024:** Updates have been made to the "PV generation system metadata.ttl" file in the Metadata folder, adding information regarding the azimuth and tilt angles of the PV modules within the PV generation systems.
