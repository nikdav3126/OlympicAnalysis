approute = "https://pro4-oly.herokuapp.com/api/v1.0/athletes"

function getInputFromBox1() {

    var gender = document.getElementById("gender").value;
    console.log("gender");

    var gender = document.getElementById("noc").value;
    console.log("noc");

    var age = document.getElementById("fage").value;
    console.log("fage");


    var weight = document.getElementById("fweight").value;
    console.log("fweight");


    var height = document.getElementById("fheight").value;
    console.log("fheight");
    (async()=>{
        var response = await fetch(
            "https://pro4-oly.herokuapp.com/api/v1.0/athletes",
            {data: JSON.stringify({gender, age, weight, height}),
            method: "POST"}
        )
    })()
}

