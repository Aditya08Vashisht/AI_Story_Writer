const state = {
    mode: 'hook',
    demos: [],
    lastOutput: '',
    staticMode: false,
};

const els = {};

const MODE_LABELS = {
    hook: 'Hook Generator',
    expand: 'Scene Expander',
    continue: 'Story Continuation',
};

const LOCAL_DEMOS = [
    {
        label: 'Audio thriller',
        mode: 'hook',
        genre: 'thriller',
        tone: 'suspenseful',
        language: 'hinglish',
        idea: 'A struggling podcast writer receives voice notes from a missing listener before each episode releases.',
        characters: 'Rhea, Kabir',
        length: 'medium',
    },
    {
        label: 'Family drama',
        mode: 'expand',
        genre: 'family drama',
        tone: 'emotional',
        language: 'hindi',
        idea: "A daughter returns home for her brother's wedding and finds her late mother's diary hidden in the prayer room.",
        characters: 'Anaya, Dev',
        length: 'medium',
    },
    {
        label: 'Romance cliffhanger',
        mode: 'continue',
        genre: 'romance',
        tone: 'cinematic',
        language: 'english',
        idea: 'Two former radio hosts meet again during a city blackout, with one final unsent confession between them.',
        characters: 'Aarav, Meera',
        length: 'long',
    },
];

const GENRE_DATA = {
    thriller: {
        sensory: 'a locked phone buzzing under the floorboards',
        stakes: 'a secret that could ruin an entire family',
        setting: 'a half-lit apartment above a silent marketplace',
        turn: 'the safest witness is the one who has been lying from the start',
    },
    romance: {
        sensory: 'a message left unsent at midnight',
        stakes: 'a love story that arrives at the worst possible time',
        setting: 'a city terrace after the first rain',
        turn: 'the person walking away has already written the ending',
    },
    horror: {
        sensory: 'a prayer bell ringing by itself',
        stakes: 'a fear that knows every name in the house',
        setting: 'an ancestral home where every mirror is covered',
        turn: 'the ghost is not asking for revenge, but for help',
    },
    fantasy: {
        sensory: 'a map that redraws itself in moonlight',
        stakes: 'a kingdom balanced on one forbidden choice',
        setting: 'a border town built around a sleeping dragon shrine',
        turn: 'the chosen one is only a decoy for the real heir',
    },
    'family drama': {
        sensory: 'a festival light flickering above an empty chair',
        stakes: 'a truth that can either heal the family or split it forever',
        setting: 'a crowded home on the morning of a wedding',
        turn: 'the person everyone blames is the only one protecting them',
    },
    comedy: {
        sensory: 'a microphone squealing at exactly the wrong moment',
        stakes: 'a tiny lie growing into a full public disaster',
        setting: 'a community hall five minutes before the chief guest arrives',
        turn: 'the worst plan in the room accidentally solves everything',
    },
};

document.addEventListener('DOMContentLoaded', () => {
    bindElements();
    bindEvents();
    checkHealth();
    loadDemoPrompts();
});

function bindElements() {
    [
        'healthStatus',
        'loadDemoBtn',
        'ideaInput',
        'genreSelect',
        'toneSelect',
        'languageSelect',
        'lengthSelect',
        'charactersInput',
        'demoPromptRow',
        'generateBtn',
        'clearBtn',
        'copyBtn',
        'modeLabel',
        'storyTitle',
        'genreMeta',
        'toneMeta',
        'languageMeta',
        'storyOutput',
        'pitchOutput',
        'bibleOutput',
        'styleOutput',
    ].forEach((id) => {
        els[id] = document.getElementById(id);
    });
}

function bindEvents() {
    document.querySelectorAll('.mode-tab').forEach((tab) => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.mode-tab').forEach((item) => item.classList.remove('active'));
            tab.classList.add('active');
            state.mode = tab.dataset.mode;
            els.modeLabel.textContent = tab.textContent.trim();
        });
    });

    els.generateBtn.addEventListener('click', generateStory);
    els.clearBtn.addEventListener('click', resetForm);
    els.copyBtn.addEventListener('click', copyOutput);
    els.loadDemoBtn.addEventListener('click', () => applyDemo(0));
}

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        els.healthStatus.textContent = data.status === 'online' ? 'Online' : 'Offline';
        els.healthStatus.classList.toggle('offline', data.status !== 'online');
    } catch (error) {
        state.staticMode = true;
        els.healthStatus.textContent = 'Static Demo';
        els.healthStatus.classList.remove('offline');
    }
}

async function loadDemoPrompts() {
    try {
        const res = await fetch('/api/demo');
        state.demos = await res.json();
        renderDemoButtons();
        if (state.demos.length) applyDemo(0);
    } catch (error) {
        state.staticMode = true;
        state.demos = LOCAL_DEMOS;
        renderDemoButtons();
        applyDemo(0);
    }
}

function renderDemoButtons() {
    els.demoPromptRow.innerHTML = state.demos.map((demo, index) => (
        `<button class="prompt-chip" type="button" data-index="${index}">${escapeHtml(demo.label)}</button>`
    )).join('');

    els.demoPromptRow.querySelectorAll('.prompt-chip').forEach((chip) => {
        chip.addEventListener('click', () => applyDemo(Number(chip.dataset.index)));
    });
}

function applyDemo(index) {
    const demo = state.demos[index];
    if (!demo) return;

    state.mode = demo.mode;
    document.querySelectorAll('.mode-tab').forEach((tab) => {
        tab.classList.toggle('active', tab.dataset.mode === demo.mode);
        if (tab.dataset.mode === demo.mode) els.modeLabel.textContent = tab.textContent.trim();
    });

    els.ideaInput.value = demo.idea;
    els.genreSelect.value = demo.genre;
    els.toneSelect.value = demo.tone;
    els.languageSelect.value = demo.language;
    els.lengthSelect.value = demo.length;
    els.charactersInput.value = demo.characters;
}

async function generateStory() {
    const payload = {
        mode: state.mode,
        idea: els.ideaInput.value,
        genre: els.genreSelect.value,
        tone: els.toneSelect.value,
        language: els.languageSelect.value,
        length: els.lengthSelect.value,
        characters: els.charactersInput.value,
    };

    els.generateBtn.disabled = true;
    els.generateBtn.textContent = 'Generating';
    els.storyOutput.textContent = 'Drafting...';

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        renderResult(data);
    } catch (error) {
        renderResult(generateLocalStory(payload));
    } finally {
        els.generateBtn.disabled = false;
        els.generateBtn.textContent = 'Generate';
    }
}

function renderResult(data) {
    state.lastOutput = data.output || '';
    els.storyTitle.textContent = data.title || 'Untitled Story';
    els.modeLabel.textContent = data.mode_label || 'Story Output';
    els.genreMeta.textContent = titleCase(data.genre);
    els.toneMeta.textContent = titleCase(data.tone);
    els.languageMeta.textContent = titleCase(data.language);
    els.storyOutput.textContent = state.lastOutput;
    els.pitchOutput.textContent = data.pitch || '';

    const bible = data.story_bible || {};
    els.bibleOutput.innerHTML = Object.entries(bible).map(([key, value]) => {
        const display = Array.isArray(value) ? value.join(', ') : value;
        return `<dt>${escapeHtml(titleCase(key.replaceAll('_', ' ')))}</dt><dd>${escapeHtml(display)}</dd>`;
    }).join('');

    els.styleOutput.innerHTML = (data.style_notes || []).map((note) => (
        `<li>${escapeHtml(note)}</li>`
    )).join('');
}

function resetForm() {
    els.ideaInput.value = '';
    els.charactersInput.value = '';
    els.genreSelect.value = 'thriller';
    els.toneSelect.value = 'cinematic';
    els.languageSelect.value = 'english';
    els.lengthSelect.value = 'medium';
    els.storyTitle.textContent = 'Ready for a story seed';
    els.storyOutput.textContent = 'Choose a mode, add a story seed, and generate a demo-ready sample.';
    els.pitchOutput.textContent = 'The pitch line will appear here.';
    els.bibleOutput.innerHTML = '<dt>Central Conflict</dt><dd>Waiting for generation</dd>';
    els.styleOutput.innerHTML = '<li>Voice, pacing, and format notes will appear after generation.</li>';
}

async function copyOutput() {
    if (!state.lastOutput) return;
    try {
        await navigator.clipboard.writeText(state.lastOutput);
        els.copyBtn.textContent = 'Copied';
        setTimeout(() => {
            els.copyBtn.textContent = 'Copy';
        }, 1200);
    } catch (error) {
        els.copyBtn.textContent = 'Select Text';
    }
}

function titleCase(value) {
    return String(value || '').replace(/\w\S*/g, (word) => (
        word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ));
}

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = String(value || '');
    return div.innerHTML;
}

function generateLocalStory(payload) {
    const genre = GENRE_DATA[payload.genre] ? payload.genre : 'thriller';
    const data = GENRE_DATA[genre];
    const characters = parseCharacters(payload.characters, payload.idea);
    const lead = characters[0];
    const second = characters[1] || 'Meera';
    const title = makeTitle(payload.idea, genre);
    const languageNote = languageWrap(payload.language);
    let output = '';

    if (payload.mode === 'hook') {
        output = `When ${lead} discovers ${data.sensory}, ${second} insists it is nothing, until the same warning appears inside tomorrow's audio script.`;
    } else if (payload.mode === 'expand') {
        output = [
            `${lead} reached ${data.setting} with the feeling that the place had been waiting. ${capitalize(data.sensory)} cut through the ordinary noise around them, turning every small movement into a warning. The idea was simple on paper: ${payload.idea}. In the room, it felt much larger, as if the story had found a body and started breathing.`,
            `${second} tried to laugh it off, but the laugh came out too thin. "Say exactly what you saw," ${second} said. ${lead} looked at the doorway, then at the shadow underneath it. The answer should have been easy. Instead, ${lead} noticed the one detail that did not belong: the scene was already arranged like the final page of a script.`,
            `The moment broke when a new sound entered the room. Not loud, not dramatic, just precise. It forced both of them to understand the same thing at once: ${data.turn}.`,
        ].join('\n\n');
    } else {
        output = [
            `${lead} does not answer immediately. The silence gives the room enough time to change. Somewhere nearby, ${data.sensory}, and ${second} realizes the old version of the plan is useless now.`,
            `Instead of explaining, ${lead} opens the last message again. The words are the same, but the meaning is not. It is no longer a warning. It is an invitation.`,
            `They decide to follow the clue before sunrise. That choice keeps the story moving and raises the central question: who benefits if ${data.stakes} stays buried?`,
        ].join('\n\n');
    }

    return {
        title,
        mode: payload.mode,
        mode_label: MODE_LABELS[payload.mode] || 'Story Output',
        genre,
        tone: payload.tone || 'cinematic',
        language: payload.language || 'english',
        output: `${output}${languageNote}`,
        pitch: `A ${payload.tone || 'cinematic'} ${genre} built around ${lead}'s choice, ${data.stakes}, and an audio-first cliffhanger.`,
        story_bible: {
            central_conflict: data.stakes,
            primary_characters: characters,
            recurring_image: data.sensory,
            next_episode_question: `What does ${lead} lose if the truth comes out now?`,
        },
        style_notes: [
            'Voice: visual, atmospheric, and ready for voice performance.',
            'Pacing: clear scene turns with a cliffhanger finish.',
            'Format: optimized for serialized listening.',
            `Language mode: ${titleCase(payload.language || 'english')}.`,
        ],
    };
}

function parseCharacters(characters, idea) {
    const source = characters || idea || '';
    const names = source.split(/[,;/]/).map((name) => name.trim()).filter(Boolean);
    if (names.length) return names.slice(0, 4);
    return ['Aarav', 'Meera'];
}

function makeTitle(idea, genre) {
    const words = String(idea || '').match(/[A-Za-z0-9]+/g) || [];
    const useful = words.filter((word) => word.length > 3).slice(0, 4);
    return useful.length ? useful.map(capitalize).join(' ') : `Untitled ${titleCase(genre)}`;
}

function languageWrap(language) {
    if (language === 'hinglish') {
        return '\n\nPerformance flavor: keep the narration English-led, but let emotional dialogue land in natural Hinglish where it feels intimate.';
    }
    if (language === 'hindi') {
        return '\n\nHindi adaptation note: preserve the beats, translate dialogue naturally, and keep narration suitable for Hindi audio drama.';
    }
    return '';
}

function capitalize(value) {
    const text = String(value || '');
    return text.charAt(0).toUpperCase() + text.slice(1);
}
