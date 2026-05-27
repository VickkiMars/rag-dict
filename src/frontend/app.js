const apiBase = ""; // Same origin

const $ = id => document.getElementById(id);

// --- State and Tab Management ---
function switchTab(tabName) {
  const tabs = ['forward', 'reverse'];
  tabs.forEach(t => {
    const btn = $(`tab-${t}-btn`);
    const content = $(`tab-${t}`);
    if (t === tabName) {
      btn.classList.add('active');
      content.classList.add('active');
    } else {
      btn.classList.remove('active');
      content.classList.remove('active');
    }
  });
  
  // Clear suggestions dropdown on tab change
  $("suggestions").innerHTML = "";
  $("suggestions").style.display = "none";
}

// --- API Calls ---
async function defineWord(word) {
  const res = await fetch(`${apiBase}/define?word=${encodeURIComponent(word)}`);
  return res.json();
}

async function reverseSearch(meaning) {
  const res = await fetch(`${apiBase}/reverse?meaning=${encodeURIComponent(meaning)}`);
  return res.json();
}

async function autocompletePrefix(prefix) {
  if (!prefix) return [];
  const res = await fetch(`${apiBase}/autocomplete?prefix=${encodeURIComponent(prefix)}`);
  if (!res.ok) return [];
  const j = await res.json();
  return j.matches || [];
}

// --- UI Logic and Listeners ---
(() => {
  const wordInput = $("wordInput");
  const defineBtn = $("defineBtn");
  const suggestionsBox = $("suggestions");
  
  const defineResultCard = $("defineResultCard");
  const resultWord = $("resultWord");
  const resultPos = $("resultPos");
  const defineLatencyBadge = $("defineLatencyBadge");
  const resultPronunciationContainer = $("resultPronunciationContainer");
  const resultPronunciation = $("resultPronunciation");
  const resultDefinitions = $("resultDefinitions");
  const resultEtymologyContainer = $("resultEtymologyContainer");
  const resultEtymology = $("resultEtymology");
  const defineSource = $("defineSource");

  const defineErrorCard = $("defineErrorCard");
  const defineErrorMsg = $("defineErrorMsg");
  const fallbackSuggestions = $("fallbackSuggestions");
  const suggestionButtons = $("suggestionButtons");

  const meaningInput = $("meaningInput");
  const reverseBtn = $("reverseBtn");
  const reverseResultCard = $("reverseResultCard");
  const reverseLatencyBadge = $("reverseLatencyBadge");
  const reverseResultsList = $("reverseResultsList");
  const reverseErrorCard = $("reverseErrorCard");
  const reverseErrorMsg = $("reverseErrorMsg");

  // Autocomplete debouncing
  let acTimer = 0;
  wordInput.addEventListener("input", () => {
    clearTimeout(acTimer);
    const val = wordInput.value.trim();
    if (!val) {
      suggestionsBox.innerHTML = "";
      suggestionsBox.style.display = "none";
      return;
    }
    
    acTimer = setTimeout(async () => {
      const matches = await autocompletePrefix(val);
      if (matches.length === 0) {
        suggestionsBox.innerHTML = "";
        suggestionsBox.style.display = "none";
        return;
      }
      
      suggestionsBox.innerHTML = "";
      suggestionsBox.style.display = "block";
      
      matches.slice(0, 10).forEach(match => {
        const btn = document.createElement("button");
        btn.textContent = match;
        btn.onclick = () => {
          wordInput.value = match;
          suggestionsBox.innerHTML = "";
          suggestionsBox.style.display = "none";
          defineBtn.click();
        };
        suggestionsBox.appendChild(btn);
      });
    }, 180);
  });

  // Hide suggestions if clicking outside
  document.addEventListener("click", (e) => {
    if (e.target !== wordInput && e.target !== suggestionsBox) {
      suggestionsBox.style.display = "none";
    }
  });

  // Execute Forward Lookup
  defineBtn.addEventListener("click", async () => {
    const word = wordInput.value.trim();
    if (!word) return;

    // Reset UI states
    defineResultCard.classList.add("hidden");
    defineErrorCard.classList.add("hidden");
    suggestionsBox.style.display = "none";

    const startTime = performance.now();
    try {
      const data = await defineWord(word);
      const latency = (performance.now() - startTime).toFixed(1);

      if (data.error) {
        // Render spelling suggestions if not found
        defineErrorMsg.textContent = data.error;
        suggestionButtons.innerHTML = "";
        
        if (data.suggestions && data.suggestions.length > 0) {
          fallbackSuggestions.classList.remove("hidden");
          data.suggestions.forEach(suggestion => {
            const btn = document.createElement("button");
            btn.textContent = suggestion;
            btn.onclick = () => {
              wordInput.value = suggestion;
              defineBtn.click();
            };
            suggestionButtons.appendChild(btn);
          });
        } else {
          fallbackSuggestions.classList.add("hidden");
        }
        defineErrorCard.classList.remove("hidden");
      } else {
        // Render dictionary entry
        const entry = data.result;
        resultWord.textContent = entry.word;
        
        // Render POS
        if (entry.pos && entry.pos.length > 0) {
          resultPos.textContent = entry.pos.join(", ");
          resultPos.classList.remove("hidden");
        } else {
          resultPos.classList.add("hidden");
        }

        // Render Pronunciations
        if (entry.pronunciations && entry.pronunciations.length > 0) {
          resultPronunciation.textContent = entry.pronunciations.join("  |  ");
          resultPronunciationContainer.classList.remove("hidden");
        } else {
          resultPronunciationContainer.classList.add("hidden");
        }

        // Render Definitions
        resultDefinitions.innerHTML = "";
        entry.definitions.forEach(def => {
          const li = document.createElement("li");
          li.textContent = def;
          resultDefinitions.appendChild(li);
        });

        // Render Etymology
        if (entry.etymology) {
          resultEtymology.textContent = entry.etymology;
          resultEtymologyContainer.classList.remove("hidden");
        } else {
          resultEtymologyContainer.classList.add("hidden");
        }

        // Set Source and Latency
        defineSource.textContent = data.source === "cache" ? "Redis Cache" : "Disk Database";
        defineLatencyBadge.textContent = `${latency} ms`;
        if (parseFloat(latency) < 15) {
          defineLatencyBadge.style.borderColor = "rgba(16, 185, 129, 0.25)";
          defineLatencyBadge.style.color = "var(--accent-green)";
          defineLatencyBadge.style.background = "rgba(16, 185, 129, 0.1)";
        } else {
          defineLatencyBadge.style.borderColor = "rgba(59, 130, 246, 0.2)";
          defineLatencyBadge.style.color = "var(--accent-blue)";
          defineLatencyBadge.style.background = "rgba(59, 130, 246, 0.1)";
        }

        defineResultCard.classList.remove("hidden");
      }
    } catch (err) {
      console.error(err);
      defineErrorMsg.textContent = "An error occurred connecting to the FastAPI dictionary server.";
      fallbackSuggestions.classList.add("hidden");
      defineErrorCard.classList.remove("hidden");
    }
  });

  // Execute Reverse Lookup
  reverseBtn.addEventListener("click", async () => {
    const meaning = meaningInput.value.trim();
    if (!meaning) return;

    // Reset UI states
    reverseResultCard.classList.add("hidden");
    reverseErrorCard.classList.add("hidden");

    const startTime = performance.now();
    try {
      const data = await reverseSearch(meaning);
      const latency = (performance.now() - startTime).toFixed(1);

      if (data.error) {
        reverseErrorMsg.textContent = data.error;
        reverseErrorCard.classList.remove("hidden");
      } else {
        const results = data.result || [];
        reverseResultsList.innerHTML = "";
        
        if (results.length === 0) {
          reverseErrorMsg.textContent = "No semantically matching terms found. Try a different phrase.";
          reverseErrorCard.classList.remove("hidden");
          return;
        }

        results.forEach(match => {
          const card = document.createElement("div");
          card.className = "semantic-match-card";
          
          const posString = match.pos && match.pos.length > 0 ? `(${match.pos.join(", ")})` : "";
          
          card.innerHTML = `
            <div class="match-header">
              <span class="match-word">${match.word}</span>
              <span class="pos-badge">${match.pos ? match.pos[0] : 'definition'}</span>
              <div class="match-score-badges">
                <span class="score-badge cosine" title="Cosine Vector Similarity">Cos: ${(match.score || 0).toFixed(3)}</span>
                <span class="score-badge bm25" title="BM25 Lexical Score">BM25: ${(match.bm25_score || 0).toFixed(3)}</span>
                <span class="score-badge combined" title="Combined Score (0.6 * Semantic + 0.4 * BM25)">Combined: ${(match.combined_score || 0).toFixed(3)}</span>
              </div>
            </div>
            <p class="match-meaning">${match.meaning}</p>
          `;
          
          // Clicking a word in reverse lookup copies it to forward lookup and queries it!
          card.onclick = () => {
            wordInput.value = match.word;
            switchTab('forward');
            defineBtn.click();
          };
          
          reverseResultsList.appendChild(card);
        });

        // Set Source and Latency
        const reverseSource = $("reverseSource");
        reverseSource.textContent = data.source === "cache" 
          ? "Redis Cache" 
          : "FAISS Vector Store + BM25 Lexical Reranker";
        
        reverseLatencyBadge.textContent = `${latency} ms`;
        if (parseFloat(latency) < 15) {
          reverseLatencyBadge.style.borderColor = "rgba(16, 185, 129, 0.25)";
          reverseLatencyBadge.style.color = "var(--accent-green)";
          reverseLatencyBadge.style.background = "rgba(16, 185, 129, 0.1)";
        } else {
          reverseLatencyBadge.style.borderColor = "rgba(59, 130, 246, 0.2)";
          reverseLatencyBadge.style.color = "var(--accent-blue)";
          reverseLatencyBadge.style.background = "rgba(59, 130, 246, 0.1)";
        }

        reverseResultCard.classList.remove("hidden");
      }
    } catch (err) {
      console.error(err);
      reverseErrorMsg.textContent = "An error occurred connecting to the FastAPI dictionary server.";
      reverseErrorCard.classList.remove("hidden");
    }
  });

  // Make switchTab available globally
  window.switchTab = switchTab;

  // Handle enter key on input fields
  wordInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      suggestionsBox.style.display = "none";
      defineBtn.click();
    }
  });

  meaningInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      reverseBtn.click();
    }
  });

})();
