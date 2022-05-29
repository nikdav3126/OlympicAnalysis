

d3.json("archive/countryMedalCount.json").then(function(data) {
    console.log(data);
  });


  function demoInfo (sample){

    d3.json("archive/countryMedalCount.json").then((data)=>{
        let ID  = data.index.Team;
        
        let result = ID.filter(sampleResult => sampleResult.index == sample);

        let resultData = result[0];
        console.log(resultData);

        d3.select("#sample-metadata").html("");

        Object.entries(resultData).forEach(([key,value]) =>{
            d3.select("#sample-metadata")
            .append("h5").text(`${key}: ${value}`);
        });
    });
}

function buildBarChart(sample){
d3.json("archive/countryMedalCount.json").then((data)=>{
        let sampleData  = data.samples;
        
        let result = sampleData.filter(sampleResult => sampleResult.id == sample);

        let resultData = result[0];

        let otu_ids = resultData.otu_ids;
        let otu_labels = resultData.otu_labels;
        let sample_values = resultData.sample_values;

        let yticks = otu_ids.slice(0, 10).map(id => `OTU ${id}`);
        let xValues = sample_values.slice(0, 10);
        let textLabels = otu_labels.slice(0, 10);

        let barChart = {
            y: yticks.reverse(),
            x: xValues.reverse(),
            text: textLabels.reverse(),
            type: "bar",
            orientation: "h"
        }
        
        let layout = {
            title:"Top Ten Belly Button Bacteria"
        }
        Plotly.newPlot("bar",[barChart],layout);
    });
 
}

function buildBubble(sample){
    d3.json("archive/countryMedalCount.json").then((data)=>{
        let sampleData  = data.samples;
        
        let result = sampleData.filter(sampleResult => sampleResult.id == sample);

        let resultData = result[0];

        let otu_ids = resultData.otu_ids;
        let otu_labels = resultData.otu_labels;
        let sample_values = resultData.sample_values;


        let bubbleChart = {
            y: sample_values,
            x: otu_ids,
            text: otu_labels,
            mode: "markers",
            marker:{
                size: sample_values,
                color: otu_ids,
                colorscale: "Agsunset"
            }
        }
        
        let layout = {
            title:"Bacteria Cultures per Sample",
            hovermode: "closest",
            xaxis: {title: "OTU ID"}
        }
        Plotly.newPlot("bubble",[bubbleChart],layout);
    });
}

function buildGauge(sample){
    d3.json("archive/countryMedalCount.json").then((data)=>{
        let sampleData  = data.samples;
        
        let result = sampleData.filter(sampleResult => sampleResult.id == sample);

        let resultData = result[0];

        let otu_ids = resultData.otu_ids;
        let otu_labels = resultData.otu_labels;
        let sample_values = resultData.sample_values;
        let wfreq = resultData.wfreq;


        var data = [
            {
              domain: { x: [0,1], y: [0,1]},
              value: [wfreq],
              title: { text: "Scrubs Per Week" },
              type: "indicator",
              mode: "gauge+number",
              indicator: "R-Base",
              gauge: {
                axis: { range: [0, 9] },
                steps: [
                  { range: [0, 1], color: "#4f2992" },
                  { range: [1, 2], color: "#7a2b9e" },
                  { range: [2, 3], color: "#972fa1" },
                  { range: [3, 4], color: "#c3389c" },
                  { range: [4, 5], color: "#e94e89" },
                  { range: [5, 6], color: "#f66e7a" },
                  { range: [6, 7], color: "#f89078" },
                  { range: [7, 8], color: "#f4b685" },
                  { range: [8, 9], color: "#eed49f" },
                ],

              }
            }
          ];
          
          var layout = { 
              title: "Belly Button Washing Frequency"
          };
          Plotly.newPlot("gauge", data, layout);
    });
}

// Initialize dashboard
function initialize(){
    var select = d3.select("#selDataset");

    //Use d3 to read data
    d3.json("archive/countryMedalCount.JSOn").then((data)=>{
        let sampleNames = data.Team;

        sampleNames.forEach((sample) =>{
            select.append("option")
            .text(sample).property("value", sample);
        });
        
        let sample1 = sampleNames[Aruba];

        //call functions to build metadata
        demoInfo(sample1);
        buildBarChart(sample1);
        buildBubble(sample1);
        buildGauge(sample1);
    });

}

//Update the dashboard
function optionChanged(item){
    demoInfo(item);
    buildBarChart(item);
    buildBubble(item);
    buildGauge(item);
}

initialize();

  

