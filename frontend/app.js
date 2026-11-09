// TUTORIAL
const fileExtension = ".png"; 
const totalImages = 5;
let currentStep = 1;

const helpBtn = document.getElementById('helpBtn');
const tutorialWrapper = document.getElementById('tutorialWrapper');
const tutorialImg = document.getElementById('tutorialImg');
const stepText = document.getElementById('stepText');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

// 1. Core Image Swapping Function
function renderStep() {
  // Dynamically points to "tutorial1.png", "tutorial2.png", etc.
  tutorialImg.src = `img/tutorial/tutorial${currentStep}${fileExtension}`;
  stepText.textContent = `Step ${currentStep} of ${totalImages}`;
  
  // Grey out arrows if at boundaries
  prevBtn.disabled = (currentStep === 1);
  nextBtn.disabled = (currentStep === totalImages);
}

// 2. Toggle Hidden/Unhidden Logic
helpBtn.addEventListener('click', () => {
  if (tutorialWrapper.style.display === 'none' || tutorialWrapper.style.display === '') {
    currentStep = 1; // Always reset to step 1 on open
    renderStep();
    tutorialWrapper.style.display = 'block'; // Unhide
  } else {
    tutorialWrapper.style.display = 'none';  // Hide
  }
});

// 3. Arrow Click Handlers
prevBtn.addEventListener('click', () => {
  if (currentStep > 1) {
    currentStep--;
    renderStep();
  }
});

nextBtn.addEventListener('click', () => {
  if (currentStep < totalImages) {
    currentStep++;
    renderStep();
  }
});

// MODEL OPTION
const singleModel = document.getElementById("single-model");
const doubleModel = document.getElementById("double-model");

const modelReset = document.getElementById("model-reset");

const form1 = document.querySelector('#input-prompt-1 .generate-form');
const form2 = document.querySelector('#input-prompt-2 .generate-form');

const max_token1 = form1.querySelector('[name="max-token1"]');
const do_sample1 = form1.querySelector('[name="do-sample1"]');
const temperature1 = form1.querySelector('[name="temperature1"]');
const top_k1 = form1.querySelector('[name="top-k1"]');
const top_p1 = form1.querySelector('[name="top-p1"]');
const repetition_penalty1 = form1.querySelector('[name="repetition-penalty1"]');
const enable_thinking1 = form1.querySelector('[name="enable-thinking1"]');

const max_token2 = form2.querySelector('[name="max-token2"]');
const do_sample2 = form2.querySelector('[name="do-sample2"]');
const temperature2 = form2.querySelector('[name="temperature2"]');
const top_k2 = form2.querySelector('[name="top-k2"]');
const top_p2 = form2.querySelector('[name="top-p2"]');
const repetition_penalty2 = form2.querySelector('[name="repetition-penalty2"]');
const enable_thinking2 = form2.querySelector('[name="enable-thinking2"]');


// Reset button 1 and 2 creation
const reset1 = document.createElement('button');
reset1.id = 'model-1-reset';
reset1.textContent = 'Reset this model';
reset1.addEventListener('click', function(){
    form1.querySelector('[name="max-token1"]').value = parseInt(50);
    form1.querySelector('[name="do-sample1"]').checked = false;
    form1.querySelector('[name="temperature1"]').value = parseFloat(0.8);
    form1.querySelector('[name="top-k1"]').value = parseInt(50);
    form1.querySelector('[name="top-p1"]').value = parseFloat(0.95);
    form1.querySelector('[name="repetition-penalty1"]').value = parseFloat(1.1);
    form1.querySelector('[name="enable-thinking1"]').checked = false;
})

const reset2 = document.createElement('button');
reset2.id = 'model-2-reset';
reset2.textContent = 'Reset this model';
reset2.addEventListener('click', function(){
    form2.querySelector('[name="max-token2"]').value = parseInt(50);
    form2.querySelector('[name="do-sample2"]').checked = false;
    form2.querySelector('[name="temperature2"]').value = parseFloat(0.8);
    form2.querySelector('[name="top-k2"]').value = parseInt(50);
    form2.querySelector('[name="top-p2"]').value = parseFloat(0.95);
    form2.querySelector('[name="repetition-penalty2"]').value = parseFloat(1.1);
    form2.querySelector('[name="enable-thinking2"]').checked = false;
})

// Swtiches from single model to double or back.
singleModel.addEventListener('click', function(){
    // change color
    singleModel.style.backgroundColor = "black";
    singleModel.style.color = "#dcdad4";
    doubleModel.style.backgroundColor = "";
    doubleModel.style.color = "black";
    modelReset.replaceChildren(reset2);
    //disable input
    document.querySelectorAll('#input-prompt-1 input').forEach(input => {
        input.disabled = true;
    });

    const defaultLayer = [0,2,4,6,8,10,12,14];
    document.querySelectorAll('#intervention-a .generate-form input').forEach(input => {
        if(input.name.startsWith("layer")){
            const layerNumber = parseInt(input.name.slice(5));
            input.checked = defaultLayer.includes(layerNumber);
        }
        else{
            input.checked = false;
        }
        input.disabled = true;
        if(input.name.startsWith("head-idxs")){
            input.checked = true;
        }
    });

    //reset input value
    form1.querySelector('[name="max-token1"]').value = parseInt(50);
    form1.querySelector('[name="do-sample1"]').checked = false;
    form1.querySelector('[name="temperature1"]').value = parseFloat(0.8);
    form1.querySelector('[name="top-k1"]').value = parseInt(50);
    form1.querySelector('[name="top-p1"]').value = parseFloat(0.95);
    form1.querySelector('[name="repetition-penalty1"]').value = parseFloat(1.1);
    form1.querySelector('[name="enable-thinking1"]').checked = false;
})

doubleModel.addEventListener('click', function(){
    doubleModel.style.backgroundColor = "black";
    doubleModel.style.color = "#dcdad4";
    singleModel.style.backgroundColor = "#dcdad4";
    singleModel.style.color = "black";
    singleModel.addEventListener("mousedown", function(){
        singleModel.style.backgroundColor = "#b6b4af";
    })
    singleModel.addEventListener("mouseup", function(){
        singleModel.style.backgroundColor = "#dcdad4";
    })
    modelReset.replaceChildren(reset1, reset2);
    document.querySelectorAll('#input-prompt-1 input').forEach(input => {
        input.disabled = false;
    });
    document.querySelectorAll('#intervention-a .generate-form input').forEach(input => {
        input.disabled = false;
    });
})

// Call single button
modelReset.replaceChildren(reset2);

// Check out of bound inputs
max_token1.addEventListener('input', function(){
    const val = parseInt(max_token1.value);
    if(isNaN(val)) return;
    if(val < 0) max_token1.value = 0;
    if(val > 100) max_token1.value = 100;
});

temperature1.addEventListener('input', function(){
    const val = parseFloat(temperature1.value);
    if(isNaN(val)) return;
    if(val < 0) temperature1.value = 0;
    if(val > 10) temperature1.value = 10;
});

top_k1.addEventListener('input', function(){
    const val = parseInt(top_k1.value);
    if(isNaN(val)) return;
    if(val < 0) top_k1.value = 0;
    if(val > 100) top_k1.value = 100;
});

top_p1.addEventListener('input', function(){
    const val = parseFloat(top_p1.value);
    if(isNaN(val)) return;
    if(val < 0) top_p1.value = 0;
    if(val > 50) top_p1.value = 50;
})

repetition_penalty1.addEventListener('input', function(){
    const val = parseFloat(repetition_penalty1.value);
    if(isNaN(val)) return;
    if(val < 0) repetition_penalty1.value = 0;
    if(val > 1) repetition_penalty1.value = 1;
})

max_token2.addEventListener('input', function(){
    const val = parseInt(max_token2.value);
    if(isNaN(val)) return;
    if(val < 0) max_token2.value = 0;
    if(val > 100) max_token2.value = 100;
});

temperature2.addEventListener('input', function(){
    const val = parseFloat(temperature2.value);
    if(isNaN(val)) return;
    if(val < 0) temperature2.value = 0;
    if(val > 10) temperature2.value = 10;
});

top_k2.addEventListener('input', function(){
    const val = parseInt(top_k2.value);
    if(isNaN(val)) return;
    if(val < 0) top_k2.value = 0;
    if(val > 100) top_k2.value = 100;
});

top_p2.addEventListener('input', function(){
    const val = parseFloat(top_p2.value);
    if(isNaN(val)) return;
    if(val < 0) top_p2.value = 0;
    if(val > 1) top_p2.value = 1;
})

repetition_penalty2.addEventListener('input', function(){
    const val = parseFloat(repetition_penalty2.value);
    if(isNaN(val)) return;
    if(val < 0) repetition_penalty2.value = 0;
    if(val > 50) repetition_penalty2.value = 50;
})

// TOGGLE SYSTEM PROMPT
const systemPrompt = document.getElementById('system-prompt');
const systemPromptDisplay = document.getElementById('system-prompt-display');

systemPrompt.addEventListener('click', function(){
    if(systemPromptDisplay.style.display == 'none'){
        systemPromptDisplay.style.display = '';
        document.getElementById('intervention-a').style.display = '';

    }
    else{
        systemPromptDisplay.style.display = 'none';
        document.getElementById('intervention-a').style.display = 'none';
    }
})


// TOGGLE INTERVENTION VISABILITY
const interventionToggleA = document.getElementById("intervention-toggle-a");
const interventionVisibilityA = document.querySelector('#intervention-a .generate-form');
interventionToggleA.addEventListener('change', function(){
    if (interventionToggleA.checked) {
        interventionVisibilityA.style.display = '';
    } 
    else {
        interventionVisibilityA.style.display = 'none';
    }
})

const interventionToggleB = document.getElementById("intervention-toggle-b");
const interventionVisibilityB = document.querySelector('#intervention-b .generate-form');
interventionToggleB.addEventListener('change', function(){
    if (interventionToggleB.checked) {
        interventionVisibilityB.style.display = '';
    } 
    else {
        interventionVisibilityB.style.display = 'none';
    }
})

// TOGGLE MLP AND ATTENTION DISPLAY
const intervention_A_Display = document.getElementById("intervention-display-a");
const attention_A_Display = document.getElementById("attention-info-a");
const mlp_A_Display = document.getElementById("mlp-info-a");

const attention_a = document.getElementById("attention-a");
const mlp_a = document.getElementById("mlp-a");

const intervention_B_Display = document.getElementById("intervention-display-b");
const attention_B_Display = document.getElementById("attention-info-b");
const mlp_B_Display = document.getElementById("mlp-info-b");

const attention_b = document.getElementById("attention-b");
const mlp_b = document.getElementById("mlp-b");

// Hides mlpDisplay
mlp_A_Display.style.display = 'none';
mlp_B_Display.style.display = 'none';

// Toggle
let type_A = "attention";
let type_B = "attention";

// Pre select attention
attention_a.style.backgroundColor = "black";
attention_a.style.color = "#dcdad4";

attention_b.style.backgroundColor = "black";
attention_b.style.color = "#dcdad4";

attention_a.addEventListener('click', function(){
    attention_a.style.backgroundColor = "black";
    attention_a.style.color = "#dcdad4";
    mlp_a.style.backgroundColor = "";
    mlp_a.style.color = "black";
    type_A = "attention";
    mlp_A_Display.style.display = "none";
    attention_A_Display.style.display = "";
})

mlp_a.addEventListener('click', function(){
    mlp_a.style.backgroundColor = "black";
    mlp_a.style.color = "#dcdad4";
    attention_a.style.backgroundColor = "#dcdad4";
    attention_a.style.color = "black";
    type_A = "mlp";
    mlp_A_Display.style.display = "";
    attention_A_Display.style.display = "none";
})

attention_b.addEventListener('click', function(){
    attention_b.style.backgroundColor = "black";
    attention_b.style.color = "#dcdad4";
    mlp_b.style.backgroundColor = "";
    mlp_b.style.color = "black";
    type_B = "attention";
    mlp_B_Display.style.display = "none";
    attention_B_Display.style.display = "";
})

mlp_b.addEventListener('click', function(){
    mlp_b.style.backgroundColor = "black";
    mlp_b.style.color = "#dcdad4";
    attention_b.style.backgroundColor = "#dcdad4";
    attention_b.style.color = "black";
    type_B = "mlp";
    mlp_B_Display.style.display = "";
    attention_B_Display.style.display = "none";
})

// RUN COMPARISON
const run = document.getElementById("run");

// Loading screen
let loadingTimer;

function startLoading() {
    const text1 = document.getElementById("model-1-text");
    const text2 = document.getElementById("model-2-text");
    let dots = 0;

    document.body.style.cursor = "wait";   // spinning cursor
    run.style.cursor = "wait";             // the button has its own cursor, so set it too
    text1.textContent = text2.textContent = "Generating";

    loadingTimer = setInterval(function () {
        dots = (dots + 1) % 4;                       // 0, 1, 2, 3, 0, ...
        const message = "Generating" + ".".repeat(dots);
        text1.textContent = message;
        text2.textContent = message;
    }, 400);
}

function stopLoading() {
    clearInterval(loadingTimer);           // stop the animation
    document.body.style.cursor = "";       // back to normal
    run.style.cursor = "";
}


run.addEventListener("click", async function(){
    // Preparing values
    let generation1 = {
        max_new_tokens: parseInt(form1.querySelector('[name="max-token1"]').value),
        do_sample: form1.querySelector('[name="do-sample1"]').checked,
        temperature: parseFloat(form1.querySelector('[name="temperature1"]').value),
        top_k: parseInt(form1.querySelector('[name="top-k1"]').value),
        top_p: parseFloat(form1.querySelector('[name="top-p1"]').value),
        repetition_penalty: parseFloat(form1.querySelector('[name="repetition-penalty1"]').value),
        enable_thinking: form1.querySelector('[name="enable-thinking1"]').checked,
    }

    let generation2;

    // Create Layers
    const layers2 = [];
    const layers1 = [];

    // Create head idxs
    const head_idxs2 = [];
    const head_idxs1 = [];

    // Create mlp info
    const proportion1 = document.querySelector('#mlp-info-a input[name="proportion"]').value;
    const seed1 = document.querySelector('#mlp-info-a input[name="seed"]').value;

    if(interventionToggleA.checked == false){
        generation2 = {
            generation:{
                max_new_tokens: parseInt(form2.querySelector('[name="max-token2"]').value),
                do_sample: form2.querySelector('[name="do-sample2"]').checked,
                temperature: parseFloat(form2.querySelector('[name="temperature2"]').value),
                top_k: parseInt(form2.querySelector('[name="top-k2"]').value),
                top_p: parseFloat(form2.querySelector('[name="top-p2"]').value),
                repetition_penalty: parseFloat(form2.querySelector('[name="repetition-penalty2"]').value),
                enable_thinking: form2.querySelector('[name="enable-thinking2"]').checked,
            }
        }
    }
    else{

        document.querySelectorAll('.layers2 input').forEach((l) => {
            if(l.checked){
                layers2.push(parseInt(l.name.slice(5)));
            }
        })

        document.querySelectorAll('.layers1 input').forEach((l) => {
            if(l.checked){
                layers1.push(parseInt(l.name.slice(5)));
            }
        })

        if(type_B === "attention"){
            document.querySelectorAll('.idxs2 input').forEach((i) => {
                if(i.checked){
                    head_idxs2.push(parseInt(i.name.slice(9)));
                }
            })

            document.querySelectorAll('.idxs1 input').forEach((i) => {
                if(i.checked){
                    head_idxs1.push(parseInt(i.name.slice(9)));
                }
            })

            generation2 = {
                generation:{
                    max_new_tokens: parseInt(form2.querySelector('[name="max-token2"]').value),
                    do_sample: form2.querySelector('[name="do-sample2"]').checked,
                    temperature: parseFloat(form2.querySelector('[name="temperature2"]').value),
                    top_k: parseInt(form2.querySelector('[name="top-k2"]').value),
                    top_p: parseFloat(form2.querySelector('[name="top-p2"]').value),
                    repetition_penalty: parseFloat(form2.querySelector('[name="repetition-penalty2"]').value),
                    enable_thinking: form2.querySelector('[name="enable-thinking2"]').checked,
                },
                intervention: {
                    type: type_B,
                    layers: layers2,
                    head_idxs: head_idxs2
                }
            }
        }
        else{
            // Get intervention value
            const proportion2 = document.querySelector('#mlp-info-b input[name="proportion"]').value;
            const seed2 = document.querySelector('#mlp-info-b input[name="seed"]').value;

            generation2 = {
                generation:{
                    max_new_tokens: parseInt(form2.querySelector('[name="max-token2"]').value),
                    do_sample: form2.querySelector('[name="do-sample2"]').checked,
                    temperature: parseFloat(form2.querySelector('[name="temperature2"]').value),
                    top_k: parseInt(form2.querySelector('[name="top-k2"]').value),
                    top_p: parseFloat(form2.querySelector('[name="top-p2"]').value),
                    repetition_penalty: parseFloat(form2.querySelector('[name="repetition-penalty2"]').value),
                    enable_thinking: form2.querySelector('[name="enable-thinking2"]').checked,
                },
                intervention: {
                    type: type_B,
                    layers: layers2,
                    proportion: parseFloat(proportion2),
                    seed: parseInt(seed2)
                }
            }
        }
    }
    const promptInput = document.getElementById('prompt-input');

    //Creating submittion JSON file
    let submit;
    if(systemPromptDisplay.style.display == 'none'){
        submit = {
            prompt: promptInput.value,
            configuration_a: {
                generation: generation1
            },
            configuration_b: generation2
        };
    }
    // Reconfigure for system prompt
    else{
        const systemPrompt1 = document.getElementById('system-prompt1');
        const systemPrompt2 = document.getElementById('system-prompt2');

        // Build intervention for A (null if its toggle is off)
        let interventionA = null;
        if(interventionToggleA.checked){
            if(type_A === "attention"){
                interventionA = {
                    type: type_A,
                    layers: layers1,
                    head_idxs: head_idxs1
                };
            }
            else{
                interventionA = {
                    type: type_A,
                    layers: layers1,
                    proportion: parseFloat(proportion1),
                    seed: parseInt(seed1)
                };
            }
        }

        // B's system prompt (empty becomes null)
        if(systemPrompt2.value === ''){
            generation2.system_prompt = null;
        }
        else{
            generation2.system_prompt = systemPrompt2.value;
        }

        //Creating submittion JSON file
        submit = {
            prompt: promptInput.value,
            configuration_a: {
                generation: generation1,
                system_prompt: systemPrompt1.value === '' ? null : systemPrompt1.value,
                intervention: interventionA
            },
            configuration_b: generation2
        };
    }
    console.log(submit);
    
    // FETCH JSON FILE
    run.disabled = true; 
    //Loading icon
    startLoading();
    //retrieves Json files
    try {
        const response = await fetch("http://127.0.0.1:8000/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(submit)
        });

        const data = await response.json();
        console.log(data);

        if (!response.ok) {
            // errors come back as { detail: [...] }
            const message = "Error: " + JSON.stringify(data.detail);
            document.getElementById("model-1-text").textContent = message;
            document.getElementById("model-2-text").textContent = message;
            return;
        }

        document.getElementById("model-1-text").textContent = data.condition_a_text;
        document.getElementById("model-2-text").textContent = data.condition_b_text;
        document.getElementById("condition-a-latency").textContent = data.condition_a_latency_ms + " ms";
        document.getElementById("condition-b-latency").textContent = data.condition_b_latency_ms + " ms";
    } catch (error) {
        console.error(error);
        document.getElementById("model-1-text").textContent = "Could not reach the backend.";
        document.getElementById("model-2-text").textContent = "Could not reach the backend.";
    } finally {
        stopLoading();
        run.disabled = false;  
    }
})

