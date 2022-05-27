const nocR = "https://oly-pro4.herokuapp.com/api/v1.0/noc-regions";

d3.json("archive/countryData.json").then(function(data) {
    console.log(data);
  });

let dropdown = $('#locality-dropdown');

dropdown.empty();

dropdown.append('<option selected="true" disabled>--Select--</option>');
dropdown.prop('selectedIndex', 0);


// Populate dropdown with list of provinces
$.getJSON(nocR, function (data) {
  $.each(data, function (key, entry) {
    dropdown.append($('<option></option>').attr('value', entry.NOC).text(entry.region));
  })
});