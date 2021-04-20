$("#btn1").click(function () {
    table();
});

$("#btn2").click(function () {
   $("#text").val('');

});


function table() { 
    var text = document.getElementById('text').value;
    if (text.length == 0) {
        alert('Please enter text or select one example!');
        window.location.reload();
        return
    };
    var data =  {"sentence": text};
    var url = 'generate_table';
    fetch(url,
        {
            method: 'POST',
            body: JSON.stringify(data),
            headers: {"Content-Type" : "application/json"}
        }
    ).then(response => {
	response.json().then(function(result) {
    	    $("#data-table").html(result);
    	    // $("#result-output").html(JSON.stringify(result));
	    // alert(JSON.stringify(result));
	    // document.getElementById('result-output').innerHTML = "TEST REPLACEMENT!";//JSON.stringify(result);
	});
    }
    ).catch(e => console.log(e));

/*
    var allTableCells = document.getElementsByTagName("td");

    for(var i = 0, max = allTableCells.length; i < max; i++) {
	var node = allTableCells[i];
	document.getElementById('result-output').innerHTML=node;
	var currentText = node.childNodes[0].nodeValue;
	if (currentText == "ARG0"):
	    node.style.backgroundColor = "red";
    }
*/
    // document.getElementById('result-output').innerHTML='de';

};

function createOutput(json) {

    return JSON.stringify(json)
}
