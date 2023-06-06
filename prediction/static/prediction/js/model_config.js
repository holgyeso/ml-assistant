document.getElementById("id_cluster_nr").style.visibility = "hidden"
document.getElementById("id_cluster_nr").parentElement.children[0].style.visibility = "hidden"

async function getColumnValues(col_name) {
    const response = await fetch("/unique-vals/" + col_name);
    const jsonData = await response.json();
    return jsonData;
}

document.getElementById("id_model").addEventListener("change", event => {

    if (event.target.value == "clustering") {

        console.log(document.getElementById("id_cluster_nr"))

        document.getElementById("id_target_column").style.visibility = "hidden"
        document.getElementById("id_target_column").parentElement.children[0].style.visibility = "hidden"
        document.getElementById("id_prediction_class").style.visibility = "hidden"
        document.getElementById("id_prediction_class").parentElement.children[0].style.visibility = "hidden"

        document.getElementById("id_cluster_nr").style.visibility = "visible"
        document.getElementById("id_cluster_nr").parentElement.children[0].style.visibility = "visible"

        document.getElementById("id_algorithm_to_use").innerHTML = `
            <option value = "kmeans">kmeans</option>
            <option value = "DBSCAN">DBSCAN</option>
        `
    }
    else {

        document.getElementById("id_target_column").style.visibility = "visible"
        document.getElementById("id_target_column").parentElement.children[0].style.visibility = "visible"

        document.getElementById("id_cluster_nr").style.visibility = "hidden"
        document.getElementById("id_cluster_nr").parentElement.children[0].style.visibility = "hidden"

        if (event.target.value == "regression") {
            document.getElementById("id_algorithm_to_use").innerHTML = `
            <option value = "linear regression">linear regression</option>
            <option value = "SVR - rbf kernel">SVR - rbf kernel</option>
        `
            document.getElementById("id_prediction_class").style.visibility = "hidden"
            document.getElementById("id_prediction_class").parentElement.children[0].style.visibility = "hidden"
        }

        if (event.target.value == "classification") {
            document.getElementById("id_algorithm_to_use").innerHTML = `
            <option value = "logistic regression">logistic regression</option>
            <option value = "decision tree">decision tree</option>
            `
            document.getElementById("id_prediction_class").style.visibility = "visible"
            document.getElementById("id_prediction_class").parentElement.children[0].style.visibility = "visible"
        }
    }
})

document.getElementById("id_algorithm_to_use").addEventListener("change", event => {
    if (event.target.value == "kmeans") {
        document.getElementById("id_cluster_nr").style.visibility = "visible"
        document.getElementById("id_cluster_nr").parentElement.children[0].style.visibility = "visible"
    } else {
        document.getElementById("id_cluster_nr").style.visibility = "hidden"
        document.getElementById("id_cluster_nr").parentElement.children[0].style.visibility = "hidden"
    }
})

document.getElementById("id_target_column").addEventListener("change", event => {
    let pred_class = ''
    getColumnValues(event.target.value).then(e => {
        e.forEach(element => {
            pred_class += '<option value="' + element + '">' + element + '</option>'
        });
        document.getElementById("id_prediction_class").innerHTML = pred_class
    })
})