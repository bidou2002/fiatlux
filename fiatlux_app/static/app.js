'use strict';
const $ = id => document.getElementById(id);
let catalog, presets, config, selected, revision = 0, tab = 'block', running = false;
let result = null, resultRevision = -1, runId = null, picture = null, scale = 'linear', imageRequest = 0;
let dragIndex = null;
const invalidInputs = new Map();

function requireValidInputs() {
  if (invalidInputs.size) throw new Error(invalidInputs.values().next().value.error);
}
const clone = value => JSON.parse(JSON.stringify(value));
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};

function message(text, error = false) {
  $('message').hidden = !text;
  $('message').textContent = text || '';
  $('message').classList.toggle('error', error);
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {
  }
  : {
    method: 'POST', headers: {
      'Content-Type': 'application/json'
    }, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Erreur du serveur');
  return data;
}

function changed() {
  revision++;
  if (config) {
    try {
      localStorage.setItem('fiatlux-draft-v1', JSON.stringify(config));
    }
    catch (_) {
    }
  }
  updateBadge();
}

function updateBadge() {
  const node = $('result-state');
  node.className = 'badge';
  if (running) {
    node.textContent = 'Calcul en cours';
    node.classList.add('running');
  }
  else if (result && revision === resultRevision) {
    node.textContent = 'À jour';
    node.classList.add('current');
  }
  else if (picture) {
    node.textContent = 'Résultats périmés';
    node.classList.add('stale');
  }
  else node.textContent = 'En attente';
}

function blockDetail(block) {
  const p = block.params;
  if (block.type === 'PlaneWave') return `${(config.spectrum.central_wavelength * 1e9).toFixed(0)} nm · ${config.spectrum.samples} λ`;
  if (block.type === 'DeformableMirror') return `${p.commands.length} modes · OPD`;
  if (block.type.endsWith('Propagator')) return p.propagation === 'fraunhofer' ? `Fraunhofer · f = ${p.focal_length} m` : `Fresnel · z = ${p.distance} m`;
  if (block.type === 'Detector') return `${p.exposure_time} s · ${p.digitize ? 'ADU' : 'électrons'}`;
  return `Plan ${p.grid}`;
}

function renderChain() {
  const container = $('chain');
  container.replaceChildren();
  $('block-count').textContent = `${config.blocks.length} BLOCS`;
  config.blocks.forEach((block, index) => {
    const node = element('div', undefined, 'chain-node');
    if (index) node.append(element('span', undefined, 'connector'));
    const card = element('div', undefined, `block${block.id === selected ? ' selected' : ''}`);
    card.draggable = true;
    card.tabIndex = 0;
    card.setAttribute('role','button');
    card.setAttribute('aria-label', `Sélectionner ${catalog[block.type].label} · ${block.id}`);
    const top = element('div', undefined, 'block-top');
    top.append(element('span', catalog[block.type].group, 'block-kind'), element('span', String(index + 1).padStart(2,'0'), 'block-index'));
    card.append(top, element('h3', catalog[block.type].label), element('span', blockDetail(block), 'block-detail'));
    const select = () => {
      selected = block.id;
      tab = 'block';
      renderChain();
      renderParameters();
    };
    card.onclick = select;
    card.onkeydown = event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    };
    card.ondragstart = () => {
      dragIndex = index;
      card.classList.add('dragging');
    };
    card.ondragend = () => {
      dragIndex = null;
      card.classList.remove('dragging');
    };
    card.ondragover = event => event.preventDefault();
    card.ondrop = event => {
      event.preventDefault();
      if (dragIndex !== null && dragIndex !== index) move(dragIndex,index);
    };
    node.append(card);
    container.append(node);
  });
}

function move(from, to) {
  if (to < 0 || to >= config.blocks.length) return;
  const [block] = config.blocks.splice(from,1);
  config.blocks.splice(to,0,block);
  changed();
  renderChain();
  renderParameters();
}

function renderLibrary() {
  const container = $('library');
  container.replaceChildren();
  const filter = $('search').value.toLowerCase();
  let group = '';
  for (const [kind, info] of Object.entries(catalog)) {
    if (!`${kind} ${info.label}`.toLowerCase().includes(filter)) continue;
    if (info.group !== group) {
      group = info.group;
      container.append(element('div',group,'library-group'));
    }
    const button = element('button', info.label,'library-item');
    button.append(element('span','+'));
    button.onclick = () => {
      const block = {
        id: `${kind.toLowerCase()}-${Date.now().toString(36)}`, type: kind, params: Object.fromEntries(Object.entries(info.fields).map(([key, field]) => [key,clone(field.default)]))
      };
      let index = config.blocks.findIndex(b => b.type === 'Detector');
      if (kind === 'PlaneWave') index = 0;
      if (kind === 'Detector' || index < 0) index = config.blocks.length;
      config.blocks.splice(index,0,block);
      selected = block.id;
      tab = 'block';
      changed();
      renderChain();
      renderParameters();
    };
    container.append(button);
  }
}

function field(parent, key, rule, object, prefix) {
  const wrap = element('div',undefined,'field');
  const label = element('label',rule.label);
  const id = `${prefix}-${key}`;
  label.htmlFor = id;
  if (rule.unit) label.append(element('span',rule.unit));
  let input;
  if (rule.kind === 'grid' || rule.kind === 'regime') {
    input = element('select');
    const choices = rule.kind === 'grid' ? Object.keys(config.grids) : ['fraunhofer','fresnel'];
    for (const value of choices) {
      const option = element('option',value);
      option.value = value;
      input.append(option);
    }
    input.value = object[key];
  }
  else if (rule.kind === 'array') {
    input = element('textarea');
    input.value = JSON.stringify(object[key]);
    input.spellcheck = false;
  }
  else {
    input = element('input');
    input.type = rule.kind === 'bool' ? 'checkbox' : 'number';
    if (input.type === 'checkbox') input.checked = object[key];
    else {
      input.step = rule.kind === 'integer' ? '1' : 'any';
      input.value = object[key] ?? '';
      if (rule.minimum !== null && rule.minimum !== undefined) input.min = rule.minimum;
      if (rule.maximum !== null && rule.maximum !== undefined) input.max = rule.maximum;
    }
  }
  input.id = id;
  if (invalidInputs.has(id)) input.value = invalidInputs.get(id).text;
  input.onchange = () => {
    try {
      let value;
      if (rule.kind === 'bool') value = input.checked;
      else if (rule.kind === 'grid' || rule.kind === 'regime') value = input.value;
      else if (rule.kind === 'array') {
        value = JSON.parse(input.value);
        if (!Array.isArray(value) || value.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw new Error('Les coefficients doivent être un tableau JSON de nombres finis, en mètres.');
      }
      else if (input.value === '' && rule.nullable) value = null;
      else {
        value = input.value === '' ? NaN : Number(input.value);
        if (!Number.isFinite(value)) throw new Error(`${rule.label} : nombre fini requis`);
      }
      invalidInputs.delete(id);
      object[key] = value;
      changed();
      renderChain();
      message('');
    }
    catch (error) {
      invalidInputs.set(id, {text: input.value, error: `${prefix} : ${error.message}`});
      changed();
      message(`${prefix} : ${error.message}`,true);
    }
  };
  wrap.append(label,input);
  parent.append(wrap);
}

function renderParameters() {
  $('block-tab').classList.toggle('active',tab === 'block');
  $('system-tab').classList.toggle('active',tab === 'system');
  const parent = $('parameters');
  parent.replaceChildren();
  if (tab === 'system') {
    const precision = element('div',undefined,'field');
    const label = element('label','Précision CPU');
    const select = element('select');
    select.id = 'precision';
    label.htmlFor = 'precision';
    for (const value of ['float64','float32']) {
      const option = element('option',value);
      option.value = value;
      select.append(option);
    }
    select.value = config.precision;
    select.onchange = () => {
      config.precision = select.value;
      changed();
    };
    precision.append(label,select);
    parent.append(precision);
    parent.append(element('div','SPECTRE · BANDE PHOTOMÉTRIQUE','section-label'));
    const rules = {
      central_wavelength: {
        label:'Longueur d’onde centrale',unit:'m'
      }, delta_wavelength:{
        label:'Largeur de bande',unit:'m'
      }, f0:{
        label:'Flux zéro de la bande',unit:'convention Band.f0'
      }, magnitude:{
        label:'Magnitude',unit:'mag'
      }, samples:{
        label:'Canaux spectraux',kind:'integer',minimum:1,maximum:8
      }
    };
    for (const [key, rule] of Object.entries(rules)) field(parent,key,rule,config.spectrum,'spectrum');
    for (const [name, grid] of Object.entries(config.grids)) {
      parent.append(element('div',`GRILLE ${name.toUpperCase()}`,'section-label'));
      for (const key of ['nx','ny','dx','dy']) field(parent,key,{
        label:{
          nx:'Pixels x',ny:'Pixels y',dx:'Pas spatial x',dy:'Pas spatial y'
        }
        [key],kind:key[0] === 'n' ? 'integer':'number',unit:key[0] === 'n' ? 'pixels':'m'
      },grid,name);
    }
    parent.append(element('p','Les coordonnées utilisent (index − taille // 2) × pas. Les grilles référencées par les blocs doivent être compatibles ; aucune correction automatique.','parameter-note'));
    return;
  }
  const block = config.blocks.find(b => b.id === selected);
  if (!block) {
    parent.append(element('p','Sélectionnez un bloc de la chaîne.'));
    return;
  }
  const info = catalog[block.type];
  const heading = element('div',undefined,'parameter-heading');
  heading.append(element('p',block.id),element('h3',info.label));
  parent.append(heading);
  for (const [key, rule] of Object.entries(info.fields)) field(parent,key,rule,block.params,block.id);
  const notes = {
    PlaneWave:'Le spectre est réglable dans « Grilles & spectre ». Le flux est normalisé par PlaneWave selon la convention FIATLUX.',DeformableMirror:'Coefficients en mètres de chemin optique (OPD), Noll 2… avec piston exclu. Le moteur fournit la base Zernike native, actuellement carrée isotrope. Pas de conversion en déplacement mécanique.',FFTPropagator:'Un seul canal spectral. La grille de sortie naturelle est λ × |f ou z| / (N × pas d’entrée). Définissez explicitement cette grille pour les éléments suivants.',MFTPropagator:'Fraunhofer par défaut. Pour Fresnel, renseignez distance et videz focale. À distance zéro, la grille de sortie doit être identique.',ELTHarmoniPupil:'Pupille analytique, sans FITS. Les exemples CPU sont sous-échantillonnés : ils ne résolvent pas les espacements millimétriques des segments.',Kolmogorov:'Un écran OPD reproductible avec graine fixée. Échelle externe vide : Kolmogorov pur ; piston retiré.'
  };
  if (notes[block.type]) parent.append(element('p',notes[block.type],'parameter-note'));
  const controls = element('div',undefined,'block-actions');
  const index = config.blocks.indexOf(block);
  for (const [text,action] of [['← Avant',()=>move(index,index-1)],['Après →',()=>move(index,index+1)],['Supprimer',()=>{
    config.blocks.splice(index,1);
      for (const key of invalidInputs.keys()) if (key.startsWith(`${block.id}-`)) invalidInputs.delete(key);
    selected=config.blocks[Math.max(0,index-1)]?.id;
    changed();
    renderChain();
    renderParameters();
  }
  ]]) {
    const button = element('button',text,text === 'Supprimer' ? 'danger' : '');
    button.onclick = action;
    if (text === '← Avant') button.disabled = index === 0;
    if (text === 'Après →') button.disabled = index === config.blocks.length-1;
    controls.append(button);
  }
  parent.append(controls);
}

function setConfig(next) {
  invalidInputs.clear();
  config = clone(next);
  selected = config.blocks[1]?.id || config.blocks[0]?.id;
  $('project-name').value = config.name;
  changed();
  renderChain();
  renderParameters();
}

function populateResults() {
  const stage = $('stage');
  stage.replaceChildren();
  result.steps.forEach((step,index) => {
    const option = element('option',`${String(index+1).padStart(2,'0')} · ${catalog[config.blocks.find(b=>b.id === step.id)?.type]?.label || step.label}`);
    option.value = index;
    stage.append(option);
  });
  stage.value = result.steps.length-1;
  stage.disabled = false;
  const channel = $('channel');
  channel.replaceChildren();
  const sum = element('option','Polychromatique · Σ Iλ');
  sum.value = 'sum';
  channel.append(sum);
  result.wavelengths.forEach((lambda,index) => {
    const option = element('option',`${(lambda*1e9).toFixed(2)} nm`);
    option.value = index;
    channel.append(option);
  });
  channel.disabled = false;
  $('quantity').disabled = false;
  $('quantity').value = 'intensity';
  adaptView();
}

function adaptView() {
  if (!result) return;
  const step = result.steps[Number($('stage').value)];
  const quantity = $('quantity');
  for (const option of quantity.options) option.disabled = option.value === 'phase' && step.detector || option.value === 'opd' && !step.opd;
  if (quantity.selectedOptions[0].disabled) quantity.value = 'intensity';
  const isPhase = quantity.value === 'phase';
  for (const option of $('channel').options) {
    option.disabled = option.value === 'sum' ? isPhase : step.detector;
  }
  if (step.detector) $('channel').value = 'sum';
  if (isPhase && $('channel').value === 'sum') $('channel').value = '0';
  $('opd-unit-wrap').hidden = quantity.value !== 'opd';
  $('log').disabled = quantity.value !== 'intensity';
  if (quantity.value !== 'intensity') scale = 'linear';
  updateScale();
  fetchImage();
}

function updateScale() {
  $('linear').classList.toggle('active',scale === 'linear');
  $('log').classList.toggle('active',scale === 'log');
}

async function fetchImage() {
  if (!result || running) return;
  const request = ++imageRequest;
  const params = new URLSearchParams({
    run:runId,step:$('stage').value,quantity:$('quantity').value,channel:$('channel').value,opd_unit:$('opd-unit').value
  });
  try {
    const data = await api(`/api/image?${params}`);
    if (request !== imageRequest) return;
    picture = data;
    $('image').dataset.run = runId;
    $('image').dataset.step = data.step;
    $('image').hidden = false;
    $('empty').hidden = true;
    $('image-unit').textContent = data.unit;
    $('range').textContent = `${data.minimum.toExponential(2)} → ${data.maximum.toExponential(2)}`;
    draw();
  }
  catch(error) {
    if (request === imageRequest) message(error.message,true);
  }
}

function draw() {
  if (!picture) return;
  const canvas = $('image');
  const width = Math.round($('plot').clientWidth*devicePixelRatio), height = Math.round($('plot').clientHeight*devicePixelRatio);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio,devicePixelRatio);
  const w = width/devicePixelRatio,h = height/devicePixelRatio;
  ctx.fillStyle='#f2f3eb';
  ctx.fillRect(0,0,w,h);
  const left=62,right=54,top=18,bottom=44;
  const ratio = picture.nx * (picture.x[1]-picture.x[0]) / (picture.ny * (picture.y[1]-picture.y[0]));
  let pw=w-left-right,ph=h-top-bottom;
  if (pw/ph > ratio) pw=ph*ratio;
  else ph=pw/ratio;
  const px=left+(w-left-right-pw)/2,py=top+(h-top-bottom-ph)/2;
  const off=document.createElement('canvas');
  off.width=picture.nx;
  off.height=picture.ny;
  const context=off.getContext('2d');
  const image=context.createImageData(off.width,off.height);
  let min=picture.minimum,max=picture.maximum;
  const symmetric=picture.quantity !== 'intensity';
  if(symmetric){
    max=Math.max(Math.abs(min),Math.abs(max));
    min=-max;
  }
  const floor=Math.max(max*1e-12,Number.MIN_VALUE);
  for(let y=0;
  y<picture.ny;
  y++)for(let x=0;
  x<picture.nx;
  x++){
    let value=picture.values[y][x],t;
    if(scale==='log'&&picture.quantity==='intensity'){
      t=(Math.log10(Math.max(value,floor))-Math.log10(floor))/(Math.log10(Math.max(max,floor))-Math.log10(floor)||1);
    }
    else t=(value-min)/(max-min||1);
    t=Math.max(0,Math.min(1,t));
    let color;
    if(symmetric){
      color=t<.5?[Math.round(45+210*t*2),Math.round(78+177*t*2),Math.round(148+107*t*2)]:[255,Math.round(255-172*(t-.5)*2),Math.round(255-190*(t-.5)*2)];
    }
    else{
      const stops=[[18,30,26],[35,82,61],[99,139,62],[207,219,100],[255,242,185]];
      const pos=t*4,a=Math.min(3,Math.floor(pos)),f=pos-a;
      color=stops[a].map((c,k)=>Math.round(c+(stops[a+1][k]-c)*f));
    }
    const offset=((picture.ny-1-y)*picture.nx+x)*4;
    image.data.set([...color,255],offset);
  }
  context.putImageData(image,0,0);
  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(off,px,py,pw,ph);
  ctx.strokeStyle='#a6afa0';
  ctx.lineWidth=1;
  ctx.strokeRect(px,py,pw,ph);
  ctx.fillStyle='#697562';
  ctx.font='9px monospace';
  ctx.textAlign='center';
  const fmt=n=>n===0?'0':Math.abs(n)<.001?n.toExponential(1):Number(n.toPrecision(3)).toString();
  for(const f of [0,.5,1]){
    let xi=Math.round(f*(picture.nx-1)),yi=Math.round(f*(picture.ny-1));
    ctx.fillText(fmt(picture.x[xi]),px+f*pw,py+ph+17);
    ctx.textAlign='right';
    ctx.fillText(fmt(picture.y[yi]),px-7,py+(1-f)*ph+3);
    ctx.textAlign='center';
  }
  ctx.fillText('x (m)',px+pw/2,py+ph+33);
  ctx.save();
  ctx.translate(px-44,py+ph/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('y (m)',0,0);
  ctx.restore();
  const gx=px+pw+16;
  const gradient=ctx.createLinearGradient(0,py+ph,0,py);
  gradient.addColorStop(0,symmetric?'#2d4e94':'#121e1a');
  gradient.addColorStop(.5,symmetric?'#fff':'#638b3e');
  gradient.addColorStop(1,symmetric?'#ff5341':'#fff2b9');
  ctx.fillStyle=gradient;
  ctx.fillRect(gx,py,8,ph);
  ctx.font='8px monospace';
  ctx.fillStyle='#697562';
  ctx.textAlign='left';
  ctx.fillText('max',gx+12,py+7);
  ctx.fillText('min',gx+12,py+ph);
  canvas.plotBounds={
    px,py,pw,ph
  };
}
$('image').onmousemove=event=>{
  if(!picture)return;
  const bounds=$('image').plotBounds;
  if(!bounds)return;
  const rect=$('image').getBoundingClientRect();
  const x=Math.floor((event.clientX-rect.left-bounds.px)/bounds.pw*picture.nx),y=picture.ny-1-Math.floor((event.clientY-rect.top-bounds.py)/bounds.ph*picture.ny);
  if(x>=0&&x<picture.nx&&y>=0&&y<picture.ny)$('coordinates').textContent=`x ${picture.x[x].toExponential(3)} m · y ${picture.y[y].toExponential(3)} m · ${picture.values[y][x].toExponential(3)} ${picture.unit}`;
};
$('run').onclick=async()=>{
  if(running)return;
  try { requireValidInputs(); } catch (error) { message(error.message, true); return; }
  running=true;
  const snapshot=clone(config),atRevision=revision;

  result=null;
  imageRequest++;
  $('run').disabled=true;
  $('run').textContent='Calcul en cours…';
  for(const id of ['stage','quantity','channel'])$(id).disabled=true;
  updateBadge();
  message('');
  try{
    const accepted=await api('/api/run',snapshot);
    runId=accepted.run_id;
    let state;
    do{
      await new Promise(resolve=>setTimeout(resolve,250));
      state=await api(`/api/status?run=${runId}`);
    }
    while(state.state==='running');
    if(state.state==='error')throw new Error(state.error);
    result=state.result;
    resultRevision=atRevision;
    running=false;
    populateResults();
    $('run-info').textContent=`Calcul réel · ${result.precision} · CPU · ${result.elapsed.toFixed(3)} s · ${result.wavelengths.length} canal(aux) · ${result.steps.length} étapes`;
    if(revision!==atRevision)message('Le calcul correspond à la configuration de départ. Vos modifications rendent ces résultats périmés.');
  }
  catch(error){
    result=null;
    message(error.message,true);
    $('run-info').textContent=picture?'Dernière carte conservée ; résultats périmés. Relancez après correction.':'Aucun résultat calculé. La configuration est conservée.';
  }
  finally{
    running=false;
    $('run').disabled=false;
    $('run').textContent='▶ Lancer la simulation';
    updateBadge();
  }
};
$('validate').onclick=async()=>{
  try{
    requireValidInputs();
    await api('/api/validate',config);
    message('Configuration compatible : paramètres, spectre et grilles vérifiés.');
  }
  catch(error){
    message(error.message,true);
  }
};
$('open').onclick=()=>$('file').click();
$('file').onchange=async()=>{
  const file=$('file').files[0];
  if(!file)return;
  try{
    if(file.size>1000000)throw new Error('Fichier supérieur à 1 Mo');
    const text=await file.text();
    const response=await fetch('/api/validate',{
      method:'POST',headers:{
        'Content-Type':'application/json'
      },body:text
    });
    const data=await response.json();
    if(!response.ok)throw new Error(data.error);
    setConfig(JSON.parse(text));
    message('Configuration chargée.');
  }
  catch(error){
    message(error.message,true);
  }
  finally{
    $('file').value='';
  }
};
$('save').onclick=async()=>{
  try{
    requireValidInputs();
    await api('/api/validate',config);
    const blob=new Blob([JSON.stringify(config,null,2)+'\n'],{
      type:'application/json'
    });
    const url=URL.createObjectURL(blob);
    const a=element('a');
    a.href=url;
    a.download='fiatlux-chain.json';
    a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
    message('Configuration enregistrée en JSON v1.');
  }
  catch(error){
    message(error.message,true);
  }
};
$('project-name').onchange=()=>{
  config.name=$('project-name').value;
  changed();
};
$('search').oninput=renderLibrary;
$('block-tab').onclick=()=>{
  tab='block';
  renderParameters();
};
$('system-tab').onclick=()=>{
  tab='system';
  renderParameters();
};
$('stage').onchange=adaptView;
$('quantity').onchange=adaptView;
$('channel').onchange=fetchImage;
$('opd-unit').onchange=fetchImage;
$('linear').onclick=()=>{
  scale='linear';
  updateScale();
  draw();
};
$('log').onclick=()=>{
  scale='log';
  updateScale();
  draw();
};
new ResizeObserver(draw).observe($('plot'));
(async()=>{
  try{
    const data=await api('/api/catalog');
    catalog=data.components;
    presets=data.examples;
    for(const [key,preset]of Object.entries(presets)){
      const option=element('option',preset.name);
      option.value=key;
      $('examples').append(option);
    }
    let initial=presets[Object.keys(presets)[0]];
    const saved=localStorage.getItem('fiatlux-draft-v1');
    if(saved){
      try{
        const draft=JSON.parse(saved);
        await api('/api/validate',draft);
        initial=draft;
      }
      catch(_){
        message('Le brouillon précédent est incompatible ; l’exemple initial a été chargé.',true);
      }
    }
    setConfig(initial);
    renderLibrary();
    $('examples').onchange=()=>{
      setConfig(presets[$('examples').value]);
      message('Exemple chargé. Lancez la simulation pour recalculer.');
    };
  }
  catch(error){
    message(`Initialisation : ${error.message}`,true);
    $('run').disabled=true;
  }
})();
