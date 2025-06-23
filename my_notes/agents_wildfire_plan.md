
## Aims


agent + Wildfire middle and post disasters

## people

## Journal

## How to do and Plans


1. We need build an agent that have memory and be called to call tools (June 1- 5)
	1. Try one agent system first, one agent call all the tools
	2. Prototype 
2. We need build all the tools (June 5-June30)
	1. We may need a warper
		1. PostGIS (SQL)
		2. QGIS
		3. GrassGIS (Bash or Command line)
		4. Whitebox Geospatial
		5. GDAL
		6. GeoDA
		7. API calls/ Functions (Like VLM)
		8. Mapping system
			1. geopandas plot
			2. geoplot
			3. cartopy
			4.  [mapclassify](https://pysal.org/mapclassify/index.html)
			5. Matplotlib
			6. Generic Mapping Tools
3. We need provide contexts to allow the agent to correctly call the tools (July 1-July 10)
	1. Promptings of the agent
	2. We need rag of the tools documents
4.  we need build the benchmark and testing (July 11 -august1)
	1. collect the dataset and questions
		1. Start with the dataset here.
		2. 
	2. I suggested first do survey of the data on specific area like Moui, Hi and then specific questions based on these data or via vice.
5. improvement (September )
	1. Promptings
	2. agent model fine-turning with RL
		1. We need RL Datasets
		2. refer example from openmanusRL
		3. use - Verl
6. Building the chatUI (September )
	1. The system only have the chatUI and all the maps showed in the chat as the agent reply
7. Draft


## questions examples

1. How large is the area currently burning
2. What is the rate of spread
3. What direction is the fire moving towards
4. How severe is the ecological damage or vegetation loss
5. **How many people need immediate evacuation?**
6. **Which communities or populated areas are currently or potentially threatened?**
7. **Which wildlife habitats and protected areas have been impacted?**
8. **Are water resources (reservoirs, rivers) affected or contaminated?**
9. **What routes are available and safe for evacuation?**
10. **Are major transportation corridors (roads, railways) accessible or compromised?**
11. **Where are emergency shelters located relative to affected areas?**
12. **Which locations might become vulnerable next?**
13. Do a suitable analysis



## Agent frameworks comments

1. Camel-AI and OWL
2. Chat and UI Framework: based on OWL





## MVP

An GIS and RS agent that could call different tools, it should natively support qwen vlm using camel AI. The first tool need to be inetrgrated is that could call tools like Qwen API tools to QA to a remote sensing image.



