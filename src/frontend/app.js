const apiBase = ""; // same origin; if backend on different port, set e.g. "http://localhost:8000"

const $ = id => document.getElementById(id);

async function defineWord(word){
  const res = await fetch(`${apiBase}/define?word=${encodeURIComponent(word)}`);
  if(!res.ok){
    const err = await res.json().catch(()=>({error:res.statusText}));
    return err;
  }
  return res.json();
}

async function reverseSearch(meaning){
  const res = await fetch(`${apiBase}/reverse?meaning=${encodeURIComponent(meaning)}`);
  if(!res.ok) return {error: res.statusText};
  return res.json();
}

async function autocompletePrefix(prefix){
  // this endpoint must be implemented server-side: /autocomplete?prefix=...
  const res = await fetch(`${apiBase}/autocomplete?prefix=${encodeURIComponent(prefix)}`);
  if(!res.ok) return [];
  const j = await res.json();
  return j.matches || [];
}

/* UI wiring */
(() => {
  const wordInput = $("wordInput");
  const defineBtn = $("defineBtn");
  const defineResult = $("defineResult");
  const suggestionsBox = $("suggestions");
  const meaningInput = $("meaningInput");
  const reverseBtn = $("reverseBtn");
  const reverseResult = $("reverseResult");

  defineBtn.addEventListener("click", async ()=>{
    const w = wordInput.value.trim();
    if(!w) { defineResult.textContent = "enter a word"; return; }
    defineResult.textContent = "loading...";
    const r = await defineWord(w);
    defineResult.textContent = JSON.stringify(r, null, 2);
  });

  reverseBtn.addEventListener("click", async ()=>{
    const q = meaningInput.value.trim();
    if(!q){ reverseResult.textContent = "enter a phrase"; return; }
    reverseResult.textContent = "loading...";
    const r = await reverseSearch(q);
    reverseResult.textContent = JSON.stringify(r, null, 2);
  });

  let acTimer = 0;
  wordInput.addEventListener("input", ()=>{
    clearTimeout(acTimer);
    const v = wordInput.value.trim();
    if(!v){ suggestionsBox.innerHTML = ""; return; }
    acTimer = setTimeout(async ()=>{
      const matches = await autocompletePrefix(v);
      suggestionsBox.innerHTML = "";
      matches.slice(0,8).forEach(m=>{
        const btn = document.createElement("button");
        btn.textContent = m;
        btn.onclick = ()=>{ wordInput.value = m; defineBtn.click(); }
        suggestionsBox.appendChild(btn);
      });
    }, 220);
  });
})();
