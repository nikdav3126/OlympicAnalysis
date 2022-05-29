approute = "https://pro4-oly.herokuapp.com/api/v1.0/athletes"

function getInputFromBox1() {

    var gender = document.getElementById("Sex").value;
    console.log("Sex");

    var gender = document.getElementById("NOC").value;
    console.log("NOC");

    var age = document.getElementById("Age").value;
    console.log("Age");
    
    var height = document.getElementById("fheight").value;
    console.log("Height");


    var weight = document.getElementById("Weight").value;
    console.log("Weight");


    (async()=>{
        var response = await fetch(
            "https://pro4-oly.herokuapp.com/api/v1.0/athletes",
            {data: JSON.stringify({Sex, Age, Height, Weight, NOC}),
            method: "POST"}
        )
    })()
}

