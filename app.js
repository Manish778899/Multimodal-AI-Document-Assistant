const $ = s => document.querySelector(s);
const clamp = (v,a=0,b=1) => Math.min(b,Math.max(a,v));
const mix = (a,b,t) => a+(b-a)*t;
const phase = (p,a,b) => clamp((p-a)/(b-a));
const smooth = v => v*v*(3-2*v);
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const journey = $('.journey');
let queued=false,lastStep=-1;
const road=$('#journey-road'),world=$('.route-world'),viewport=$('.route-viewport');
const roadLength=road.getTotalLength();
const stops=[...document.querySelectorAll('.route-stop')];
const milestones=[
 {year:'2016',title:'A little truck.<br><em>A big appetite.</em>',copy:'It started in Meerut. One food truck, bold chaap, and a reason to come back.',status:'THE FIRST MILE'},
 {year:'2018',title:'A bend in the road.<br><em>A place of our own.</em>',copy:'The truck finds a home. Our first restaurant opens its doors, and a new chapter begins.',status:'WELCOME TO OUR FIRST RESTAURANT'},
 {year:'2021',title:'Meet. Believe.<br><em>Build together.</em>',copy:'The owner steps out to meet new partners. Conversations become plans. Plans become new neighbourhood favourites.',status:'NEW PEOPLE. SHARED AMBITION.'},
 {year:'2025',title:'More people.<br><em>More places. More swaad.</em>',copy:'From one little truck to a family of 40+ outlets. The next stop could be your city.',status:'THE JOURNEY CONTINUES'}
];
function setStep(step){if(lastStep===step)return;lastStep=step;const m=milestones[step];$('#year').textContent=m.year;$('#journey-title').innerHTML=m.title;$('#journey-copy').textContent=m.copy;$('#route-status').textContent=m.status;document.querySelectorAll('.timeline button').forEach((b,i)=>{b.classList.toggle('active',i===step);b.setAttribute('aria-pressed',String(i===step));});}
const keyframes=[[0,0],[.29,.28],[.40,.28],[.52,.44],[.58,.44],[.70,.61],[.76,.61],[.88,.82],[.92,.82],[1,1]];
function routeAt(p){for(let i=1;i<keyframes.length;i++){const [endP,endR]=keyframes[i], [startP,startR]=keyframes[i-1];if(p<=endP)return mix(startR,endR,smooth(phase(p,startP,endP)));}return 1;}
function point(r){return road.getPointAtLength(clamp(r)*roadLength);}
function renderStory(p){
 const r=routeAt(p),pos=point(r),prev=point(Math.max(0,r-.002)),next=point(Math.min(1,r+.002));
 const scale=viewport.clientWidth/1000,worldHeight=1700*scale;
 const camera=clamp(pos.y*scale-viewport.clientHeight*.58,0,Math.max(0,worldHeight-viewport.clientHeight));
 world.style.height=`${worldHeight}px`;world.style.transform=`translateY(${-camera}px)`;
 const facing=next.x>=prev.x?1:-1;const angle=clamp(Math.atan2(next.y-prev.y,Math.abs(next.x-prev.x))*180/Math.PI,-22,22);
 const truck=$('.truck-traveller'),owner=$('.owner-traveller');
 for(const el of [truck,owner]){el.style.left=`${pos.x/10}%`;el.style.top=`${pos.y/17}%`;}
 truck.style.opacity=1-phase(p,.34,.40);
 truck.style.transform=`translate(-50%,-83%) rotate(${facing*angle}deg)`;
 truck.querySelector('img').style.transform=`scaleX(${facing})`;
 const walking=p>.4&&p<.99;owner.style.opacity=phase(p,.39,.43);
 owner.style.transform=`translate(-50%,calc(-92% - ${walking?Math.abs(Math.sin(r*roadLength*.13))*3:0}px))`;
 owner.querySelector('img').style.transform=`scaleX(${facing})`;
 stops.forEach((el,i)=>{const sr=+el.dataset.route,pt=point(sr);el.style.left=`${pt.x/10}%`;el.style.top=`${pt.y/17}%`;const near=Math.abs(r-sr);el.classList.toggle('met',r>=sr-.008);el.classList.toggle('meeting',near<.022);if(i===0){el.querySelector('.restaurant-open-state').style.opacity=phase(p,.30,.35);} });
 $('.road-travelled').style.strokeDasharray=`${roadLength*r} ${roadLength}`;
 $('.journey-distance b').style.width=`${p*100}%`;
 setStep(p<.29?0:p<.4?1:p<.8?2:3);
}
function update(){queued=false;const max=document.documentElement.scrollHeight-innerHeight;$('#progress').style.width=`${max?scrollY/max*100:0}%`;if(!reduced.matches)renderStory(clamp(-journey.getBoundingClientRect().top/(journey.offsetHeight-innerHeight)));}
addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(update);}},{passive:true});addEventListener('resize',update);
reduced.addEventListener('change',()=>location.reload());
document.querySelectorAll('.timeline button').forEach(b=>b.addEventListener('click',()=>{const step=+b.dataset.step,p=[.13,.35,.55,.9][step];if(reduced.matches){renderStory(p);return;}scrollTo({top:journey.getBoundingClientRect().top+scrollY+(journey.offsetHeight-innerHeight)*p,behavior:'smooth'});}));
const dialog=$('#enquiry');
$('#enquire').addEventListener('click',()=>dialog.showModal());
$('.close').addEventListener('click',()=>dialog.close());
$('#done').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
$('#demo-form').addEventListener('submit',e=>{e.preventDefault();$('#demo-form').hidden=true;$('#demo-result').hidden=false;});
dialog.addEventListener('close',()=>{$('#demo-form').reset();$('#demo-form').hidden=false;$('#demo-result').hidden=true;});
renderStory(0);update();






