
function send_to(where, text){
    textarea = gradioApp().querySelector('#beautifulprompt_selected_text textarea')
    textarea.value = text
    updateInput(textarea)

    gradioApp().querySelector('#beautifulprompt_send_to_'+where).click()

    where == 'txt2img' ? switch_to_txt2img() : switch_to_img2img()
}

function send_to_txt2img(text){ send_to('txt2img', text) }
function send_to_img2img(text){ send_to('img2img', text) }

// function send_to(where, text){
//     textarea = gradioApp().querySelector(`#${where}_prompt textarea`)
//     textarea.value = text
//     updateInput(textarea)

//     // gradioApp().querySelector('#beautifulprompt_send_to_'+where).click()

//     where == 'txt2img' ? switch_to_txt2img() : switch_to_img2img()
// }

function send_to_txt2img(text){ send_to('txt2img', text) }
function send_to_img2img(text){ send_to('img2img', text) }

function submit_prompt(){
    var id = randomId()
    requestProgress(id, gradioApp().getElementById('beautifulprompt_results_column'), null, function(){})

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}

function show_or_hide(model_selection){
    element = gradioApp().querySelector('#beautifulprompt_api')
    if (model_selection === 'API') {
        element.style.display = 'flex'
    } else {
        element.style.display = 'none'
    }
}

