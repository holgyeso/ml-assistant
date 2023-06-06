const selects = document.getElementsByTagName("select")
let messages_ul = document.getElementById("messages")

async function getColumnValues(col_name) {
    const response = await fetch("/unique-vals/" + col_name);
    if (response.status == 400) {
        messages_ul.innerHTML += '<li class="messages--error">There is missing data in the column, therefore it cannot be JSON serialized.</li>'
    }
    else {
        const jsonData = await response.json();
        return jsonData;
    }

}

Array.from(selects).forEach(element => {
    element.addEventListener("change", (event) => {

        let name_id = event.target.id.split("-")
        name_id[name_id.length - 1] = "field_name"

        let field_name = document.getElementById(name_id.join("-")).value
        const form_id = name_id[1]

        if (event.target.value == "ordinal - sklearn.OrdinalEncoder") {

            let div_data = '<div><p>Please specify the encodings:</p>'

            let option_list = ''

            getColumnValues(field_name).then(elem => {

                for (let index = 0; index < elem.length; index++) {
                    option_list += '<option value="' + index + '">' + index + "</option>"

                }

                elem.forEach(e => {
                    let replaced_e = e.replaceAll(" ", "")
                    replaced_e = replaced_e.replaceAll("-", "")
                    console.log(replaced_e)
                    div_data += '<div><label for="id_form-' + form_id + '-ordinal-' + replaced_e + '">' + replaced_e + '</label><select name="form-' + form_id + '-ordinal-' + replaced_e + '" id="id_form-' + form_id + '-ordinal-' + replaced_e + '">' + option_list + '</select></div>'
                })

                div_data += '</div>'

                event.target.parentElement.nextElementSibling.innerHTML = div_data
            })
        }
        else {
            event.target.parentElement.nextElementSibling.innerHTML = "<div></div>"
            messages_ul.innerHTML = ""
        }
    })
});

document.getElementById("feature_form").addEventListener("submit", event => {
    event.preventDefault()
    prevent_default = false
    Array.from(selects).forEach(element => {

        let name_id = element.id.split("-")

        if (name_id[2] != "ordinal") {
            name_id[name_id.length - 1] = "field_name"

            let field_name = document.getElementById(name_id.join("-")).value

            name_id[name_id.length - 1] = "dtype"
            let field_dtype = document.getElementById(name_id.join("-")).value

            name_id[name_id.length - 1] = "include"
            let include = document.getElementById(name_id.join("-")).checked

            if (include && field_dtype == "object" && (element.value == "numerical - standard_scaler" || element.value == "numerical - min_max" || element.value == "numerical - no encoding")) {
                prevent_default = true
                messages_ul.innerHTML += '<li class="messages--error">' + field_name + ' is of type object, therefore it cannot be encoded with numerical standardization.</li>'
            }

            if (include && element.value == "ordinal - sklearn.OrdinalEncoder") {
                let used_nrs = []
                Array.from(element.parentElement.parentElement.children[1].children[0].children).forEach(e => {
                    if (e.children.length > 0) {
                        if (used_nrs.includes(e.children[1].value)) {
                            prevent_default = true
                        }
                        else {
                            used_nrs.push(e.children[1].value)
                        }
                    }
                })
                if (prevent_default) 
                    messages_ul.innerHTML += '<li class="messages--error">' + field_name + ': there are duplicate encodings in the order!</li>'
            }

        }
    })
    if (!prevent_default) {
        document.getElementById("feature_form").submit()
    }
})