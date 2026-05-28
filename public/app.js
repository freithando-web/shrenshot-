const STAGES = [
  {
    icon: '💧',
    name: 'Clean Water Well',
    desc: 'Drilling and equipping a borehole well to provide clean drinking water for an entire village of ~500 people.',
    goal: 8000,
  },
  {
    icon: '⚽',
    name: 'Sports Ground',
    desc: 'Building a levelled sports field with goalposts and basic equipment so children can play safely.',
    goal: 35000,
  },
  {
    icon: '🏫',
    name: 'New School Building',
    desc: 'Constructing a 4-classroom school block with desks, chairs, blackboards, and textbooks for 120 students.',
    goal: 150000,
  },
  {
    icon: '🏡',
    name: 'Community Centre',
    desc: 'A multi-purpose community hub for adult education, healthcare clinics, and youth programmes.',
    goal: 300000,
  },
];

let currentUser = null;

async function init() {
  try {
    const res = await fetch('/api/me');
    if (res.ok) {
      currentUser = await res.json();
      showApp();
    }
  } catch {}
}

function showApp() {
  document.getElementById('authOverlay').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  document.getElementById('navUser').textContent = `Hi, ${currentUser.name} 👋`;
  loadStats();
  setInterval(loadStats, 30000);
}

function showTab(tab) {
  document.getElementById('loginForm').classList.toggle('hidden', tab !== 'login');
  document.getElementById('registerForm').classList.toggle('hidden', tab !== 'register');
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab')[tab === 'login' ? 0 : 1].classList.add('active');
}

async function doLogin(e) {
  e.preventDefault();
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPassword').value;
  const errEl = document.getElementById('loginError');
  errEl.textContent = '';
  const res = await fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) { errEl.textContent = data.error; return; }
  currentUser = { name: data.name };
  showApp();
}

async function doRegister(e) {
  e.preventDefault();
  const name = document.getElementById('regName').value;
  const email = document.getElementById('regEmail').value;
  const password = document.getElementById('regPassword').value;
  const errEl = document.getElementById('registerError');
  errEl.textContent = '';
  const res = await fetch('/api/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  });
  const data = await res.json();
  if (!res.ok) { errEl.textContent = data.error; return; }
  currentUser = { name: data.name };
  showApp();
}

async function doLogout() {
  await fetch('/api/logout', { method: 'POST' });
  currentUser = null;
  document.getElementById('app').classList.add('hidden');
  document.getElementById('authOverlay').classList.remove('hidden');
  document.getElementById('loginEmail').value = '';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginError').textContent = '';
  showTab('login');
}

async function doDonate(e) {
  e.preventDefault();
  const amount = parseFloat(document.getElementById('donateAmount').value);
  const message = document.getElementById('donateMessage').value.trim();
  const errEl = document.getElementById('donateError');
  const successEl = document.getElementById('donateSuccess');
  errEl.textContent = '';
  successEl.textContent = '';

  const res = await fetch('/api/donate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount, message }),
  });
  const data = await res.json();
  if (!res.ok) { errEl.textContent = data.error; return; }

  successEl.textContent = `Thank you for your $${amount.toLocaleString()} donation! Every dollar counts. 💛`;
  document.getElementById('donateAmount').value = '';
  document.getElementById('donateMessage').value = '';
  document.querySelectorAll('.quick-amounts button').forEach(b => b.classList.remove('selected'));
  loadStats();
}

function setAmount(val) {
  document.getElementById('donateAmount').value = val;
  document.querySelectorAll('.quick-amounts button').forEach(b => {
    b.classList.toggle('selected', parseInt(b.textContent.replace('$', '')) === val);
  });
}

function scrollToDonate() {
  document.getElementById('donateSection').scrollIntoView({ behavior: 'smooth' });
}

async function loadStats() {
  const res = await fetch('/api/stats');
  const data = await res.json();
  renderStats(data);
}

function renderStats({ total, count, recent }) {
  document.getElementById('totalAmount').textContent = '$' + Math.round(total).toLocaleString();
  document.getElementById('totalDonors').textContent = count.toLocaleString();

  const finalGoal = STAGES[STAGES.length - 1].goal;
  const pct = Math.min(100, (total / finalGoal) * 100);
  document.getElementById('overallBar').style.width = pct.toFixed(1) + '%';
  document.getElementById('overallText').textContent =
    `$${Math.round(total).toLocaleString()} raised of $${finalGoal.toLocaleString()} goal (${pct.toFixed(1)}%)`;

  renderStages(total);
  renderRecent(recent);
}

function renderStages(total) {
  const grid = document.getElementById('stagesGrid');
  grid.innerHTML = '';

  let prevGoal = 0;
  STAGES.forEach((stage, i) => {
    let status, pct, progressAmount;

    if (total >= stage.goal) {
      status = 'completed';
      pct = 100;
    } else if (total > prevGoal) {
      status = 'active';
      progressAmount = total - prevGoal;
      const range = stage.goal - prevGoal;
      pct = Math.min(100, (progressAmount / range) * 100);
    } else {
      status = 'locked';
      pct = 0;
    }

    const stageAmount = stage.goal - prevGoal;
    const card = document.createElement('div');
    card.className = `stage-card ${status}`;
    card.innerHTML = `
      <div class="stage-status-badge">
        ${status === 'completed' ? '✅ Unlocked' : status === 'active' ? '🔥 In Progress' : '🔒 Locked'}
      </div>
      <div class="stage-badge">${stage.icon}</div>
      <div class="stage-num">Stage ${i + 1}</div>
      <div class="stage-name">${stage.name}</div>
      <div class="stage-desc">${stage.desc}</div>
      <div class="stage-goal">$${stage.goal.toLocaleString()}</div>
      <div class="stage-goal-label">total goal</div>
      <div class="stage-progress-wrap">
        <div class="stage-progress-bar" style="width: ${pct.toFixed(1)}%"></div>
      </div>
      <div style="font-size:12px;color:#999;margin-top:6px">${pct.toFixed(0)}% of this stage funded</div>
    `;
    grid.appendChild(card);
    prevGoal = stage.goal;
  });
}

function renderRecent(recent) {
  const list = document.getElementById('recentList');
  if (!recent.length) {
    list.innerHTML = '<div class="no-donations">No donations yet — be the first! 💛</div>';
    return;
  }
  list.innerHTML = recent.map(d => {
    const initial = d.user_name.charAt(0).toUpperCase();
    const time = new Date(d.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    return `
      <div class="recent-item">
        <div class="recent-avatar">${initial}</div>
        <div class="recent-info">
          <div class="recent-name">${escHtml(d.user_name)}</div>
          ${d.message ? `<div class="recent-msg">"${escHtml(d.message)}"</div>` : ''}
          <span class="recent-time">${time}</span>
        </div>
        <div class="recent-amount">$${Math.round(d.amount).toLocaleString()}</div>
      </div>
    `;
  }).join('');
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

init();
