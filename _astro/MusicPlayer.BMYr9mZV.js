import{o as _a,b as ya,i as wa,a as V,e as gt,s as et}from"./props.D0f53uoi.js";import{a0 as ka,a9 as Ea,aG as Fa,aH as Aa,aI as De,u as Le,aJ as $a,aK as Se,aL as Ca,V as Te,aM as yt,a7 as Da,aN as La,k as wt,i as ct,a as x,p as Sa,b as Ta,s as d,m as L,ah as K,g as t,f as j,d as p,c as n,r as o,n as Ma,t as X,j as F}from"./utils.CHuniVsO.js";import{a as Ba,s as st}from"./render.BejrcOws.js";import{i as _}from"./if.ORsbxKfw.js";import{e as Ra,i as ja}from"./each.CYgB9iNB.js";import{I as y}from"./Icon.DtaE14Df.js";import{I as za}from"./zh_TW.D48XCwBq.js";import{i as Ia}from"./translation.DOLBC1V9.js";import"./siteConfig.BkWEP7Yh.js";import"./navBarConfig.kp_eansV.js";const Pa=()=>performance.now(),nt={tick:l=>requestAnimationFrame(l),now:()=>Pa(),tasks:new Set};function Me(){const l=nt.now();nt.tasks.forEach(i=>{i.c(l)||(nt.tasks.delete(i),i.f())}),nt.tasks.size!==0&&nt.tick(Me)}function Na(l){let i;return nt.tasks.size===0&&nt.tick(Me),{promise:new Promise(c=>{nt.tasks.add(i={c:l,f:c})}),abort(){nt.tasks.delete(i)}}}function Rt(l,i){Se(()=>{l.dispatchEvent(new CustomEvent(i))})}function Oa(l){if(l==="float")return"cssFloat";if(l==="offset")return"cssOffset";if(l.startsWith("--"))return l;const i=l.split("-");return i.length===1?i[0]:i[0]+i.slice(1).map(c=>c[0].toUpperCase()+c.slice(1)).join("")}function Ae(l){const i={},c=l.split(";");for(const A of c){const[w,f]=A.split(":");if(!w||f===void 0)break;const z=Oa(w.trim());i[z]=f.trim()}return i}const Wa=l=>l;function Ya(l,i,c,A){var w=(l&$a)!==0,f="both",z,Z=i.inert,lt=i.style.overflow,k,$;function tt(){return Se(()=>z??=c()(i,A?.()??{},{direction:f}))}var G={is_global:w,in(){i.inert=Z,Rt(i,"introstart"),k=Jt(i,tt(),$,1,()=>{Rt(i,"introend"),k?.abort(),k=z=void 0,i.style.overflow=lt})},out(T){i.inert=!0,Rt(i,"outrostart"),$=Jt(i,tt(),k,0,()=>{Rt(i,"outroend"),T?.()})},stop:()=>{k?.abort(),$?.abort()}},at=ka;if((at.transitions??=[]).push(G),Ba){var m=w;if(!m){for(var v=at.parent;v&&(v.f&Ea)!==0;)for(;(v=v.parent)&&(v.f&Fa)===0;);m=!v||(v.f&Aa)!==0}m&&De(()=>{Le(()=>G.in())})}}function Jt(l,i,c,A,w){var f=A===1;if(Ca(i)){var z,Z=!1;return Te(()=>{if(!Z){var T=i({direction:f?"in":"out"});z=Jt(l,T,c,A,w)}}),{abort:()=>{Z=!0,z?.abort()},deactivate:()=>z.deactivate(),reset:()=>z.reset(),t:()=>z.t()}}if(c?.deactivate(),!i?.duration)return w(),{abort:yt,deactivate:yt,reset:yt,t:()=>A};const{delay:lt=0,css:k,tick:$,easing:tt=Wa}=i;var G=[];if(f&&c===void 0&&($&&$(0,1),k)){var at=Ae(k(0,1));G.push(at,at)}var m=()=>1-A,v=l.animate(G,{duration:lt,fill:"forwards"});return v.onfinish=()=>{v.cancel();var T=c?.t()??1-A;c?.abort();var h=A-T,Y=i.duration*Math.abs(h),I=[];if(Y>0){var U=!1;if(k)for(var rt=Math.ceil(Y/16.666666666666668),B=0;B<=rt;B+=1){var J=T+h*tt(B/rt),H=Ae(k(J,1-J));I.push(H),U||=H.overflow==="hidden"}U&&(l.style.overflow="hidden"),m=()=>{var ut=v.currentTime;return T+h*tt(ut/Y)},$&&Na(()=>{if(v.playState!=="running")return!1;var ut=m();return $(ut,1-ut),!0})}v=l.animate(I,{duration:Y,fill:"forwards"}),v.onfinish=()=>{m=()=>A,$?.(A,1-A),w()}},{abort:()=>{v&&(v.cancel(),v.effect=null,v.onfinish=yt)},deactivate:()=>{w=yt},reset:()=>{A===0&&$?.(1,0)},t:()=>m()}}function $e(l,i){return l===i||l?.[La]===i}function Ce(l={},i,c,A){return De(()=>{var w,f;return Da(()=>{w=f,f=[],Le(()=>{l!==c(...f)&&(i(l,...f),w&&$e(c(...w),l)&&i(null,...w))})}),()=>{Te(()=>{f&&$e(c(...f),l)&&i(null,...f)})}}),l}function Ha(l){const i=l-1;return i*i*i+1}function qa(l,{delay:i=0,duration:c=400,easing:A=Ha,axis:w="y"}={}){const f=getComputedStyle(l),z=+f.opacity,Z=w==="y"?"height":"width",lt=parseFloat(f[Z]),k=w==="y"?["top","bottom"]:["left","right"],$=k.map(h=>`${h[0].toUpperCase()}${h.slice(1)}`),tt=parseFloat(f[`padding${$[0]}`]),G=parseFloat(f[`padding${$[1]}`]),at=parseFloat(f[`margin${$[0]}`]),m=parseFloat(f[`margin${$[1]}`]),v=parseFloat(f[`border${$[0]}Width`]),T=parseFloat(f[`border${$[1]}Width`]);return{delay:i,duration:c,easing:A,css:h=>`overflow: hidden;opacity: ${Math.min(h*20,1)*z};${Z}: ${h*lt}px;padding-${k[0]}: ${h*tt}px;padding-${k[1]}: ${h*G}px;margin-${k[0]}: ${h*at}px;margin-${k[1]}: ${h*m}px;border-${k[0]}-width: ${h*v}px;border-${k[1]}-width: ${h*T}px;min-${Z}: 0`}}const Va={local:{playlist:[{id:1,title:"使一颗心免于哀伤",artist:"知更鸟 / HOYO-MiX / Chevy",cover:"/assets/music/cover/109951169585655912.jpg",url:"/assets/music/使一颗心免于哀伤-哼唱.wav",duration:240}]},behavior:{defaultVolume:.7,defaultShuffle:!1,defaultRepeat:2,position:{bottom:16,right:16}},ui:{animation:{coverRotation:{enable:!0,speed:3}},playlist:{maxHeight:384,width:320}},errorHandling:{errorDisplayDuration:3e3}};var Ka=j('<div class="fixed bottom-20 right-4 z-[60] max-w-sm"><div class="bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-slide-up"><!> <span class="text-sm flex-1"> </span> <button class="text-white/80 hover:text-white transition-colors"><!></button></div></div>'),Xa=j("<img/>"),Ga=j('<div class="w-full h-full bg-gradient-to-br from-[var(--primary)] to-[var(--primary)]/70 flex items-center justify-center"><!></div>'),Ua=j('<div class="absolute inset-0 bg-black/20 flex items-center justify-center"><div class="w-4 h-4 bg-white/90 rounded-full flex items-center justify-center"><div class="w-2 h-2 bg-[var(--primary)] rounded-full"></div></div></div>'),Ja=j("<button><!></button>"),Qa=j('<button class="btn-plain w-10 h-10 rounded-lg"><!></button>'),Za=j('<button class="btn-plain w-10 h-10 rounded-lg"><!></button>'),tr=j("<button><!></button>"),er=j('<button class="btn-plain w-8 h-8 rounded-lg"><!></button> <div class="flex-1 h-2 bg-[var(--btn-regular-bg)] rounded-full cursor-pointer" role="slider" tabindex="0" aria-label="音量控制" aria-valuemin="0" aria-valuemax="100"><div class="h-full bg-[var(--primary)] rounded-full transition-all duration-100"></div></div>',1),ar=j("<button><!></button>"),rr=j('<span class="text-sm text-[var(--content-meta)]"></span>'),ir=j('<div class="w-6 h-6 flex items-center justify-center"><!></div>'),or=j('<span class="ml-2"> </span>'),nr=j('<div role="button" tabindex="0"><!> <div class="w-10 h-10 rounded-lg overflow-hidden bg-[var(--btn-regular-bg)] flex-shrink-0"><img class="w-full h-full object-cover"/></div> <div class="flex-1 min-w-0"><div> </div> <div> <!></div></div></div>'),sr=j('<div class="playlist-panel float-panel fixed bottom-20 right-4 overflow-hidden z-50"><div class="playlist-header flex items-center justify-between p-4 border-b border-[var(--line-divider)]"><h3 class="text-lg font-semibold text-90"> </h3> <button class="btn-plain w-8 h-8 rounded-lg"><!></button></div> <div class="playlist-content overflow-y-auto max-h-80 scrollbar-custom"></div></div>'),lr=j(`<!> <div><div style="width: 90px; height: 80px; background-color: var(--card-bg); "><div class="flex items-center gap-2 h-full"><div class="cover-container relative w-12 h-12 rounded-full overflow-hidden cursor-pointer flex-shrink-0" tabindex="0" role="button"><!> <!></div> <button class="expand-btn w-8 h-8 rounded-full btn-regular border border-[var(--line-divider)] hover:border-[var(--primary)] active:scale-95 transition-all duration-200 flex items-center justify-center flex-shrink-0" tabindex="0" aria-label="展开音乐播放器"><!></button></div></div> <div style="width: 320px; background-color: var(--card-bg);"><div class="flex items-center gap-4 mb-4"><div class="cover-container relative w-16 h-16 rounded-full overflow-hidden flex-shrink-0"><img alt="封面"/></div> <div class="flex-1 min-w-0"><div class="song-title text-lg font-bold text-90 truncate mb-1"> </div> <div class="song-artist text-sm text-50 truncate"> </div> <div class="text-xs text-30 mt-1"> </div></div> <div class="flex items-center gap-1"><button class="btn-plain w-8 h-8 rounded-lg flex items-center justify-center"><!></button></div></div> <div class="progress-section mb-4"><div class="progress-bar flex-1 h-2 bg-[var(--btn-regular-bg)] rounded-full cursor-pointer" role="slider" tabindex="0" aria-label="播放进度" aria-valuemin="0" aria-valuemax="100"><div class="h-full bg-[var(--primary)] rounded-full transition-all duration-100"></div></div></div> <div class="controls flex items-center justify-center gap-2 mb-4"><!> <!> <button><!></button> <!> <!></div> <div class="bottom-controls flex items-center gap-2"><!> <!></div></div> <!></div> <style>@keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes musicWave {
        0%, 100% { transform: scaleY(0.5); }
        50% { transform: scaleY(1); }
    }
    .music-player.collapsed-mode {
        width: 96px;
        height: 80px;
    }
    .music-player {
        max-width: 320px;
        -webkit-user-select: none;
           -moz-user-select: none;
                user-select: none;
    }
    .expanded-player {
        width: 320px;
        position: absolute;
        bottom: 0;
        right: 0;
    }
    .collapsed-player {
        position: absolute;
        bottom: 0;
        right: 0;
        backdrop-filter: blur(12px);
        border: 1px solid var(--line-divider) !important;
    }
    .expanded-player {
        border: 1px solid var(--line-divider) !important;
    }
    .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    .progress-section div:hover,
    .bottom-controls > div:hover {
        transform: scaleY(1.2);
        transition: transform 0.2s ease;
    }
    @media (max-width: 768px) {
        .music-player {
            max-width: 280px;
            /*left: 8px !important;*/
            bottom: 8px !important;
            right: 8px !important;
        }
        .music-player.expanded {
            width: calc(100vw - 16px);
            max-width: none;
            /*left: 8px !important;*/
            right: 8px !important;
        }
        .playlist-panel {
            width: calc(100vw - 16px) !important;
            /*left: 8px !important;*/
            right: 8px !important;
            max-width: none;
        }
        .controls {
            gap: 8px;
        }
        .controls button {
            width: 36px;
            height: 36px;
        }
        .controls button:nth-child(3) {
            width: 44px;
            height: 44px;
        }
    }
    @media (max-width: 480px) {
        .music-player {
            max-width: 260px;
        }
        .song-title {
            font-size: 14px;
        }
        .song-artist {
            font-size: 12px;
        }
        .controls {
            gap: 6px;
            margin-bottom: 12px;
        }
        .controls button {
            width: 32px;
            height: 32px;
        }
        .controls button:nth-child(3) {
            width: 40px;
            height: 40px;
        }
        .playlist-item {
            padding: 8px 12px;
        }
        .playlist-item .w-10 {
            width: 32px;
            height: 32px;
        }
    }
    @keyframes slide-up {
        from {
            transform: translateY(100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    .animate-slide-up {
        animation: slide-up 0.3s ease-out;
    }
    @media (hover: none) and (pointer: coarse) {
        .music-player button,
        .playlist-item {
            min-height: 44px;
        }
        .progress-section > div,
        .bottom-controls > div:nth-child(2) {
            height: 12px;
        }
    }
    /* 自定义旋转动画，停止时保持当前位置 */
    @keyframes spin-continuous {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    .cover-container img {
        animation: spin-continuous 3s linear infinite;
        animation-play-state: paused;
    }
    .cover-container img.spinning {
        animation-play-state: running;
    }
    .cover-container img:hover {
        animation-play-state: var(--rotation-pause-hover, running);
    }
    /* 让主题色按钮更有视觉反馈 */
    button.bg-\\[var\\(--primary\\)\\] {
        box-shadow: 0 0 0 2px var(--primary);
        border: none;
    }
    /* 播放列表自定义滚动条样式 */
    .scrollbar-custom {
        scrollbar-width: thin;
        scrollbar-color: rgba(156, 163, 175, 0.3) transparent;
    }
    .scrollbar-custom::-webkit-scrollbar {
        width: 6px;
    }
    .scrollbar-custom::-webkit-scrollbar-track {
        background: transparent;
        border-radius: 3px;
    }
    .scrollbar-custom::-webkit-scrollbar-thumb {
        background-color: rgba(156, 163, 175, 0.3);
        border-radius: 3px;
        -webkit-transition: background-color 0.2s ease;
        transition: background-color 0.2s ease;
    }
    .scrollbar-custom::-webkit-scrollbar-thumb:hover {
        background-color: rgba(156, 163, 175, 0.5);
    }
    .scrollbar-custom::-webkit-scrollbar-thumb:active {
        background-color: rgba(156, 163, 175, 0.7);
    }
    /* 暗色模式下的滚动条样式 */
    :global(.dark) .scrollbar-custom {
        scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
    }
    :global(.dark) .scrollbar-custom::-webkit-scrollbar-thumb {
        background-color: rgba(255, 255, 255, 0.2);
    }
    :global(.dark) .scrollbar-custom::-webkit-scrollbar-thumb:hover {
        background-color: rgba(255, 255, 255, 0.3);
    }
    :global(.dark) .scrollbar-custom::-webkit-scrollbar-thumb:active {
        background-color: rgba(255, 255, 255, 0.4);
    }</style>`,1);function xr(l,i){Ta(i,!1);const c=Va;let A=c.local?.playlist??[],w=c.behavior?.defaultVolume,f=c.behavior?.defaultShuffle,z=c.behavior?.defaultRepeat,Z=c.behavior?.position?.bottom,lt=c.behavior?.position?.right,k=c.ui?.playlist?.maxHeight,$=c.ui?.playlist?.width,tt=c.ui?.animation?.coverRotation?.enable,G=c.ui?.animation?.coverRotation?.speed,at=c.errorHandling?.errorDisplayDuration,m=L(!1),v=L(!1),T=L(!0),h=L(!1),Y=L(0),I=L(0),U=L(w),rt=L(!1),B=L(!1),J=L(f),H=L(z),ut=L(""),kt=L(!1),P=L({title:"Loading ...",artist:"Loading ...",cover:"/favicon/favicon-light-192.png",url:"",duration:0}),b=L([]),C=L(0),r=L(),ht=L(),Et=L();function jt(){!t(r)||!t(P).url||(t(m)?t(r).pause():t(r).play())}function zt(){d(T,!t(T)),t(T)?(d(v,!1),d(h,!1)):(d(v,!0),d(h,!1))}function Qt(){d(h,!t(h))}function Be(){d(J,!t(J))}function Re(){d(H,(t(H)+1)%3)}function je(){if(t(b).length===0)return;const s=t(C)>0?t(C)-1:t(b).length-1;At(s)}function Ft(){if(t(b).length===0)return;let s;if(t(J))do s=Math.floor(Math.random()*t(b).length);while(s===t(C)&&t(b).length>1);else s=t(C)<t(b).length-1?t(C)+1:0;console.log("nextSong 调用",{currentIndex:t(C),newIndex:s,playlistLength:t(b).length,isShuffled:t(J)}),At(s)}function At(s){if(s<0||s>=t(b).length)return;const q=t(m);d(C,s),t(r)&&t(r).pause(),Zt(t(b)[t(C)]),(q||!t(m))&&setTimeout(()=>{t(r)&&(t(r).readyState>=2?t(r).play().catch(()=>{}):t(r).addEventListener("canplay",()=>{t(r).play().catch(()=>{})},{once:!0}))},100)}function It(s){return s.startsWith("http://")||s.startsWith("https://")||s.startsWith("/")?s:`/${s}`}function Zt(s){!s||!t(r)||(d(P,{...s}),s.url?(d(B,!0),t(r).pause(),K(r,t(r).currentTime=0),d(Y,0),d(I,s.duration??0),t(r).removeEventListener("loadeddata",te),t(r).removeEventListener("error",ee),t(r).removeEventListener("loadstart",ae),t(r).addEventListener("loadeddata",te,{once:!0}),t(r).addEventListener("error",ee,{once:!0}),t(r).addEventListener("loadstart",ae,{once:!0}),K(r,t(r).src=It(s.url)),t(r).load()):d(B,!1))}function te(){d(B,!1),t(r)?.duration&&t(r).duration>1&&(d(I,Math.floor(t(r).duration)),t(b)[t(C)]&&K(b,t(b)[t(C)].duration=t(I)),K(P,t(P).duration=t(I)))}function ee(s){d(B,!1),Pt(`无法播放 "${t(P).title}"，正在尝试下一首...`),t(b).length>1?setTimeout(()=>Ft(),1e3):t(b).length<=1&&Pt("播放列表中没有可用的歌曲")}function ae(){}function Pt(s){d(ut,s),d(kt,!0),setTimeout(()=>{d(kt,!1)},at)}function ze(){d(kt,!1)}function Ie(s){if(!t(r)||!t(ht))return;const q=t(ht).getBoundingClientRect(),$t=(s.clientX-q.left)/q.width*t(I);K(r,t(r).currentTime=$t),d(Y,$t)}function Pe(s){if(!t(r)||!t(Et))return;const q=t(Et).getBoundingClientRect(),dt=Math.max(0,Math.min(1,(s.clientX-q.left)/q.width));d(U,dt),K(r,t(r).volume=t(U)),d(rt,t(U)===0)}function re(){t(r)&&(d(rt,!t(rt)),K(r,t(r).muted=t(rt)))}function Nt(s){if(!Number.isFinite(s)||s<0)return"0:00";const q=Math.floor(s/60),dt=Math.floor(s%60);return`${q}:${dt.toString().padStart(2,"0")}`}function Ne(){t(r)&&(t(r).addEventListener("play",()=>{d(m,!0)}),t(r).addEventListener("pause",()=>{d(m,!1)}),t(r).addEventListener("timeupdate",()=>{d(Y,t(r).currentTime)}),t(r).addEventListener("ended",()=>{console.log("歌曲播放结束",{isRepeating:t(H),currentIndex:t(C),playlistLength:t(b).length,isShuffled:t(J)}),t(H)===1?(console.log("单曲循环：重新播放当前歌曲"),K(r,t(r).currentTime=0),t(r).play().catch(()=>{})):t(H)===2?(console.log("列表循环：播放下一首"),Ft()):t(C)<t(b).length-1||t(J)?(console.log("非循环模式：播放下一首"),Ft()):(console.log("非循环模式：播放列表结束，停止播放"),d(m,!1))}),t(r).addEventListener("error",s=>{d(B,!1)}),t(r).addEventListener("stalled",()=>{}),t(r).addEventListener("waiting",()=>{}))}_a(()=>{d(r,new Audio),K(r,t(r).volume=t(U)),K(r,t(r).muted=t(rt)),Ne(),d(b,[...A]),t(b).length>0?Zt(t(b)[0]):Pt("本地播放列表为空")}),ya(()=>{t(r)&&(t(r).pause(),K(r,t(r).src=""))}),wa();var ie=wt(),Oe=ct(ie);{var We=s=>{var q=lr(),dt=ct(q);{var $t=e=>{var a=Ka(),u=n(a),E=n(u);y(E,{icon:"material-symbols:error",class:"text-xl flex-shrink-0"});var M=p(E,2),S=n(M,!0);o(M);var g=p(M,2),Q=n(g);y(Q,{icon:"material-symbols:close",class:"text-lg"}),o(g),o(u),o(a),X(()=>st(S,t(ut))),F("click",g,ze),x(e,a)};_(dt,e=>{t(kt)&&e($t)})}var Ct=p(dt,2);let oe;var Dt=n(Ct);let ne;var se=n(Dt),vt=n(se),le=n(vt);{var Ye=e=>{var a=Xa();let u;X(E=>{et(a,"src",t(P).cover),et(a,"alt",`${t(P).title??""} - ${t(P).artist??""}`),u=V(a,1,"w-full h-full object-cover transition-transform duration-300",null,u,E),gt(a,`animation-duration: ${G}s;`)},[()=>({spinning:t(m)&&!t(B)&&tt,"animate-pulse":t(B)})]),x(e,a)},He=e=>{var a=Ga(),u=n(a);y(u,{icon:"fa6-solid:music",class:"text-white text-lg"}),o(a),x(e,a)};_(le,e=>{t(P).cover?e(Ye):e(He,!1)})}var qe=p(le,2);{var Ve=e=>{var a=Ua();x(e,a)};_(qe,e=>{t(m)&&e(Ve)})}o(vt);var Lt=p(vt,2),Ke=n(Lt);y(Ke,{icon:"fa6-solid:chevron-left",class:"text-[var(--primary)] text-sm"}),o(Lt),o(se),o(Dt);var St=p(Dt,2);let ue;var Ot=n(St),Tt=n(Ot),Wt=n(Tt);let de;o(Tt);var Yt=p(Tt,2),Ht=n(Yt),Xe=n(Ht,!0);o(Ht);var qt=p(Ht,2),Ge=n(qt,!0);o(qt);var ce=p(qt,2),Ue=n(ce);o(ce),o(Yt);var ve=p(Yt,2),Vt=n(ve),Je=n(Vt);y(Je,{icon:"material-symbols:expand-more",class:"text-lg"}),o(Vt),o(ve),o(Ot);var Kt=p(Ot,2),ft=n(Kt),Qe=n(ft);o(ft),Ce(ft,e=>d(ht,e),()=>t(ht)),o(Kt);var Xt=p(Kt,2),fe=n(Xt);{var Ze=e=>{var a=Ja();let u;var E=n(a);y(E,{icon:"material-symbols:shuffle",class:"text-lg"}),o(a),X(M=>{u=V(a,1,"w-10 h-10 rounded-lg",null,u,M),a.disabled=t(b).length<=1},[()=>({"btn-regular":t(J),"btn-plain":!t(J)})]),F("click",a,Be),x(e,a)};_(fe,e=>{e(Ze)})}var pe=p(fe,2);{var ta=e=>{var a=Qa(),u=n(a);y(u,{icon:"material-symbols:skip-previous",class:"text-xl"}),o(a),X(()=>a.disabled=t(b).length<=1),F("click",a,je),x(e,a)};_(pe,e=>{e(ta)})}var pt=p(pe,2);let me;var ea=n(pt);{var aa=e=>{y(e,{icon:"eos-icons:loading",class:"text-xl"})},ra=e=>{var a=wt(),u=ct(a);{var E=S=>{y(S,{icon:"material-symbols:pause",class:"text-xl"})},M=S=>{y(S,{icon:"material-symbols:play-arrow",class:"text-xl"})};_(u,S=>{t(m)?S(E):S(M,!1)},!0)}x(e,a)};_(ea,e=>{t(B)?e(aa):e(ra,!1)})}o(pt);var be=p(pt,2);{var ia=e=>{var a=Za(),u=n(a);y(u,{icon:"material-symbols:skip-next",class:"text-xl"}),o(a),X(()=>a.disabled=t(b).length<=1),F("click",a,Ft),x(e,a)};_(be,e=>{e(ia)})}var oa=p(be,2);{var na=e=>{var a=tr();let u;var E=n(a);{var M=g=>{y(g,{icon:"material-symbols:repeat-one",class:"text-lg"})},S=g=>{var Q=wt(),D=ct(Q);{var O=R=>{y(R,{icon:"material-symbols:repeat",class:"text-lg"})},N=R=>{y(R,{icon:"material-symbols:repeat",class:"text-lg opacity-50"})};_(D,R=>{t(H)===2?R(O):R(N,!1)},!0)}x(g,Q)};_(E,g=>{t(H)===1?g(M):g(S,!1)})}o(a),X(g=>u=V(a,1,"w-10 h-10 rounded-lg",null,u,g),[()=>({"btn-regular":t(H)>0,"btn-plain":t(H)===0})]),F("click",a,Re),x(e,a)};_(oa,e=>{e(na)})}o(Xt);var ge=p(Xt,2),he=n(ge);{var sa=e=>{var a=er(),u=ct(a),E=n(u);{var M=D=>{y(D,{icon:"material-symbols:volume-off",class:"text-lg"})},S=D=>{var O=wt(),N=ct(O);{var R=it=>{y(it,{icon:"material-symbols:volume-down",class:"text-lg"})},Mt=it=>{y(it,{icon:"material-symbols:volume-up",class:"text-lg"})};_(N,it=>{t(U)<.5?it(R):it(Mt,!1)},!0)}x(D,O)};_(E,D=>{t(rt)||t(U)===0?D(M):D(S,!1)})}o(u);var g=p(u,2),Q=n(g);o(g),Ce(g,D=>d(Et,D),()=>t(Et)),X(()=>{et(g,"aria-valuenow",t(U)*100),gt(Q,`width: ${t(U)*100}%`)}),F("click",u,re),F("click",g,Pe),F("keydown",g,D=>{(D.key==="Enter"||D.key===" ")&&(D.preventDefault(),D.key==="Enter"&&re())}),x(e,a)};_(he,e=>{e(sa)})}var la=p(he,2);{var ua=e=>{var a=ar();let u;var E=n(a);y(E,{icon:"material-symbols:queue-music",class:"text-lg"}),o(a),X(M=>u=V(a,1,"btn-plain w-8 h-8 rounded-lg",null,u,M),[()=>({"text-[var(--primary)]":t(h)})]),F("click",a,Qt),x(e,a)};_(la,e=>{e(ua)})}o(ge),o(St);var da=p(St,2);{var ca=e=>{var a=sr(),u=n(a),E=n(u),M=n(E,!0);o(E);var S=p(E,2),g=n(S);y(g,{icon:"material-symbols:close",class:"text-lg"}),o(S),o(u);var Q=p(u,2);Ra(Q,5,()=>t(b),ja,(D,O,N)=>{var R=nr();let Mt;var it=n(R);{var va=W=>{var ot=ir(),xt=n(ot);{var _t=mt=>{y(mt,{icon:"material-symbols:graphic-eq",class:"text-[var(--primary)] animate-pulse"})},ba=mt=>{var Ee=wt(),ga=ct(Ee);{var ha=bt=>{y(bt,{icon:"material-symbols:pause",class:"text-[var(--primary)]"})},xa=bt=>{var Fe=rr();Fe.textContent=N+1,x(bt,Fe)};_(ga,bt=>{N===t(C)?bt(ha):bt(xa,!1)},!0)}x(mt,Ee)};_(xt,mt=>{N===t(C)&&t(m)?mt(_t):mt(ba,!1)})}o(ot),x(W,ot)};_(it,W=>{W(va)})}var Gt=p(it,2),xe=n(Gt);o(Gt);var _e=p(Gt,2),Bt=n(_e);let ye;var fa=n(Bt,!0);o(Bt);var Ut=p(Bt,2);let we;var ke=n(Ut),pa=p(ke);{var ma=W=>{var ot=or(),xt=n(ot);o(ot),X(_t=>st(xt,`(${_t??""})`),[()=>Nt(t(O).duration)]),x(W,ot)};_(pa,W=>{t(O).duration>0&&W(ma)})}o(Ut),o(_e),o(R),X((W,ot,xt,_t)=>{Mt=V(R,1,"playlist-item flex items-center gap-3 p-3 hover:bg-[var(--btn-plain-bg-hover)] cursor-pointer transition-colors",null,Mt,W),et(R,"aria-label",`播放 ${t(O).title??""} - ${t(O).artist??""}`),et(xe,"src",ot),et(xe,"alt",t(O).title),ye=V(Bt,1,"font-medium truncate",null,ye,xt),st(fa,t(O).title),we=V(Ut,1,"text-sm text-[var(--content-meta)] truncate",null,we,_t),st(ke,`${t(O).artist??""} `)},[()=>({"bg-[var(--btn-plain-bg)]":N===t(C),"text-[var(--primary)]":N===t(C)}),()=>It(t(O).cover),()=>({"text-[var(--primary)]":N===t(C),"text-90":N!==t(C)}),()=>({"text-[var(--primary)]":N===t(C)})]),F("click",R,()=>At(N)),F("keydown",R,W=>{(W.key==="Enter"||W.key===" ")&&(W.preventDefault(),At(N))}),x(D,R)}),o(Q),o(a),X(D=>{gt(a,`width: ${$}px; max-height: ${k}px;`),st(M,D)},[()=>Ia(za.playlist)]),F("click",S,Qt),Ya(3,a,()=>qa,()=>({duration:300,axis:"y"})),x(e,a)};_(da,e=>{t(h)&&e(ca)})}o(Ct),Ma(2),X((e,a,u,E,M,S,g,Q)=>{oe=V(Ct,1,"music-player fixed z-[1001] transition-all duration-300 ease-in-out",null,oe,e),gt(Ct,`bottom: ${Z}px; right: ${lt}px; ; --rotation-speed: ${G}s; --rotation-pause-hover: paused;`),ne=V(Dt,1,"collapsed-player card-base bg-[var(--float-panel-bg)] dark:bg-zinc-800/90 dark:backdrop-blur-md dark:border dark:border-zinc-700/50 rounded-2xl p-3 transition-all duration-500 ease-in-out",null,ne,a),et(vt,"aria-label",t(m)?"暂停音乐":"播放音乐"),ue=V(St,1,"expanded-player card-base bg-[var(--float-panel-bg)] dark:bg-zinc-800/90 dark:backdrop-blur-md dark:border dark:border-zinc-700/50 rounded-2xl p-4 transition-all duration-500 ease-in-out",null,ue,u),et(Tt,"title",`${t(P).title??""} - ${t(P).artist??""}`),et(Wt,"src",E),de=V(Wt,1,"w-full h-full object-cover transition-transform duration-300",null,de,M),gt(Wt,`animation-duration: ${G}s;`),st(Xe,t(P).title),st(Ge,t(P).artist),st(Ue,`${S??""} / ${g??""}`),et(ft,"aria-valuenow",t(I)>0?t(Y)/t(I)*100:0),gt(Qe,`width: ${t(I)>0?t(Y)/t(I)*100:0}%`),me=V(pt,1,"btn-regular w-12 h-12 rounded-full",null,me,Q),pt.disabled=t(B)},[()=>({expanded:t(v),"collapsed-mode":t(T)}),()=>({"opacity-0":!t(T),"scale-95":!t(T),"pointer-events-none":!t(T)}),()=>({"opacity-0":!t(v),"scale-95":!t(v),"pointer-events-none":!t(v)}),()=>It(t(P).cover),()=>({spinning:t(m)&&!t(B)&&tt,"animate-pulse":t(B)}),()=>Nt(t(Y)),()=>Nt(t(I)),()=>({"opacity-50":t(B)})]),F("click",vt,jt),F("keydown",vt,e=>{(e.key==="Enter"||e.key===" ")&&(e.preventDefault(),jt())}),F("click",Lt,zt),F("keydown",Lt,e=>{(e.key==="Enter"||e.key===" ")&&(e.preventDefault(),zt())}),F("click",Vt,zt),F("click",ft,Ie),F("keydown",ft,e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault(),t(ht).getBoundingClientRect();const u=.5*t(I);t(r)&&(K(r,t(r).currentTime=u),d(Y,u))}}),F("click",pt,jt),x(s,q)};_(Oe,s=>{s(We)})}x(l,ie),Sa()}export{xr as default};
