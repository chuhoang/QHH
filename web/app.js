const state = {
  data: null,
  classId: "",
  cameraId: "",
  seats: [],
  students: {},
  ai: null,
  aiActive: false,
  aiChanging: false,
  snapshotMode: false,
  waitingForSnapshot: false,
  drawing: false,
  dragStart: null,
  draft: null,
  pollTimer: null,
  management: { classrooms: [], cameras: [], students: [] },
  activeView: "monitor",
};

const $ = (id) => document.getElementById(id);

const DESK_COLORS = [
  "#ffb020", // amber
  "#35c8ff", // cyan
  "#c77dff", // violet
  "#45e58f", // emerald
  "#ff6b8a", // rose
  "#f5e642", // yellow
  "#4f8cff", // blue
  "#ff7a45", // orange
  "#57e3d2", // turquoise
  "#d6ff63", // lime
  "#ef72d8", // magenta
  "#9d8cff", // lavender
  "#00b8a9", // teal
  "#e56b2f", // burnt orange
  "#79a8ff", // cornflower
  "#ff9ed2", // pink
  "#8ed14f", // leaf
  "#d99bff", // lilac
  "#00d4ff", // electric cyan
  "#ffcc80", // apricot
];

function deskColor(deskNumber) {
  const index = Math.max(0, Number(deskNumber || 1) - 1) % DESK_COLORS.length;
  return DESK_COLORS[index];
}

function textColorFor(background) {
  const hex = background.replace("#", "");
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return (r * 299 + g * 587 + b * 114) / 1000 > 150 ? "#08100b" : "#ffffff";
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.className = "toast", 2800);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function option(value, label, selected = false) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  node.selected = selected;
  return node;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[character]);
}

async function loadManagement() {
  state.management = await api("/api/management");
  renderClassManagement();
  renderCameraManagement();
  renderPeopleManagement();
}

function className(classId) {
  return state.management.classrooms.find(item => item.id === classId)?.name || "Chưa chọn lớp";
}

function showView(view, updateHash = true) {
  if (!["monitor", "classes", "cameras", "people"].includes(view)) view = "monitor";
  state.activeView = view;
  document.querySelectorAll(".app-view").forEach(node => {
    node.classList.toggle("active", node.id === `${view}View`);
  });
  document.querySelectorAll(".nav-item").forEach(node => {
    node.classList.toggle("active", node.dataset.view === view);
  });
  if (updateHash && location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
  if (view === "monitor") setTimeout(resizeOverlay, 0);
  else loadManagement().catch(error => toast(error.message, true));
}

function renderClassManagement() {
  const box = $("classGrid");
  const classes = state.management.classrooms || [];
  box.classList.toggle("single", classes.length === 1);
  if (!classes.length) {
    box.innerHTML = `<div class="empty-management">Chưa có lớp học. Tạo lớp đầu tiên để bắt đầu.</div>`;
    return;
  }
  box.innerHTML = classes.map((classroom, index) => `
    <article class="entity-card ${index === 0 ? "featured" : ""}">
      <span class="entity-index">${String(index + 1).padStart(2, "0")} / ${String(classes.length).padStart(2, "0")}</span>
      <h2>${escapeHtml(classroom.name)}</h2>
      <div class="entity-meta">${escapeHtml(classroom.num_desks)} bàn · ${classroom.student_count || 0} người học</div>
      <div class="entity-actions">
        <button class="button primary" data-action="monitor-class" data-id="${classroom.id}">Mở giám sát</button>
        <button class="button secondary" data-action="edit-class" data-id="${classroom.id}">Chỉnh sửa</button>
      </div>
    </article>
  `).join("");
}

function renderCameraManagement() {
  const box = $("cameraGrid");
  const cameras = state.management.cameras || [];
  box.classList.toggle("single", cameras.length === 1);
  if (!cameras.length) {
    box.innerHTML = `<div class="empty-management">Chưa có camera. Thêm thiết bị để nhận luồng hình ảnh.</div>`;
    return;
  }
  box.innerHTML = cameras.map((camera, index) => {
    const stream = state.data?.cameras?.find(item => item.id === camera.id);
    const tags = (camera.class_ids || []).map(id => `<span class="class-tag">${escapeHtml(className(id))}</span>`).join("");
    return `
      <article class="entity-card ${index === 0 ? "featured" : ""}">
        <span class="entity-index">${escapeHtml(camera.ipAddress || "Không có IP")} · ${escapeHtml(camera.brand || "Camera")}</span>
        <h2>${escapeHtml(camera.name)}</h2>
        <div class="status-line"><span class="dot ${stream?.stream_ready ? "ok" : "bad"}"></span>${stream?.stream_ready ? "Đã có video record" : "Chưa có video record"}</div>
        <div class="class-tags">${tags || `<span class="class-tag">Chưa gắn lớp</span>`}</div>
        <div class="entity-actions">
          <button class="button primary" data-action="monitor-camera" data-id="${camera.id}">Xem camera</button>
          <button class="button secondary" data-action="edit-camera" data-id="${camera.id}">Cấu hình</button>
        </div>
      </article>
    `;
  }).join("");
}

function renderPeopleManagement() {
  const filter = $("peopleClassFilter");
  const selected = filter.value;
  filter.innerHTML = "";
  filter.append(option("", "Tất cả lớp", !selected));
  for (const classroom of state.management.classrooms || []) {
    filter.append(option(classroom.id, classroom.name, classroom.id === selected));
  }
  const query = $("peopleSearch").value.trim().toLowerCase();
  const people = (state.management.students || []).filter(person => {
    const matchesClass = !filter.value || person.class_id === filter.value;
    const haystack = `${person.name} ${person.student_code}`.toLowerCase();
    return matchesClass && (!query || haystack.includes(query));
  });
  $("peopleList").innerHTML = people.length ? people.map(person => `
    <article class="person-row">
      ${person.has_face
        ? `<img class="person-face" src="/api/students/face?id=${encodeURIComponent(person.id)}" alt="Khuôn mặt ${escapeHtml(person.name)}">`
        : `<span class="person-mark">${escapeHtml((person.name || "?").trim().slice(0, 1).toUpperCase())}</span>`}
      <div><div class="person-name">${escapeHtml(person.name)}</div><div class="person-sub">${escapeHtml(person.student_code)}</div></div>
      <div class="person-class">${escapeHtml(className(person.class_id))}</div>
      <div class="person-sub">${person.desk_num ? `Bàn ${person.desk_num} · Chỗ ${person.slot_num || 1}` : "Chưa xếp chỗ"} · ${person.has_face ? "Đã có ảnh mặt" : "Chưa có ảnh mặt"}</div>
      <div class="entity-actions"><button class="button secondary" data-action="edit-person" data-id="${person.id}">Chỉnh sửa</button></div>
    </article>
  `).join("") : `<div class="empty-management">Không tìm thấy người học phù hợp.</div>`;
}

function openEditor(type, item = {}) {
  const panel = $("editorPanel");
  const form = $("editorForm");
  panel.classList.add("open");
  panel.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";

  if (type === "class") {
    $("editorEyebrow").textContent = item.id ? "Cập nhật lớp học" : "Lớp học mới";
    $("editorTitle").textContent = item.id ? item.name : "Thêm lớp học";
    form.innerHTML = `
      <input type="hidden" name="entity" value="class"><input type="hidden" name="id" value="${escapeHtml(item.id || "")}">
      <label><span>Tên lớp</span><input name="name" required value="${escapeHtml(item.name || "")}" placeholder="Ví dụ: 10 Chuyên Hóa 1"></label>
      <label><span>Số bàn</span><input name="num_desks" type="number" min="1" max="100" required value="${escapeHtml(item.num_desks || 20)}"></label>
      <p class="form-note">Sau khi tạo lớp, hãy gắn camera và xếp người học vào bàn tương ứng.</p>
      ${formActions()}
    `;
  } else if (type === "camera") {
    const checked = new Set(item.class_ids || []);
    const classChecks = (state.management.classrooms || []).map(classroom => `
      <label class="check-option"><input type="checkbox" name="class_ids" value="${classroom.id}" ${checked.has(classroom.id) ? "checked" : ""}>${escapeHtml(classroom.name)}</label>
    `).join("");
    $("editorEyebrow").textContent = item.id ? "Cấu hình thiết bị" : "Nguồn hình ảnh mới";
    $("editorTitle").textContent = item.id ? item.name : "Thêm camera";
    form.innerHTML = `
      <input type="hidden" name="entity" value="camera"><input type="hidden" name="id" value="${escapeHtml(item.id || "")}">
      <div class="form-grid">
        ${field("name", "Tên camera", item.name, "Camera cửa lớp", true, "wide")}
        ${field("ipAddress", "Địa chỉ IP", item.ipAddress, "192.168.1.100", true)}
        ${field("port", "Cổng RTSP", item.port || 554, "", false, "", "number")}
        ${field("username", "Tài khoản", item.username)}
        ${field("password", "Mật khẩu", item.password, "", false, "", "password")}
        ${field("rtspPath", "Đường dẫn RTSP", item.rtspPath || item.notes || "/cam/realmonitor?channel=1&subtype=0", "", true, "wide")}
        ${field("brand", "Hãng", item.brand || "Dahua")}
        ${field("location", "Vị trí", item.location)}
      </div>
      <label><span>Gắn với lớp</span><div class="check-grid">${classChecks || `<span class="form-note">Hãy tạo lớp trước khi gắn camera.</span>`}</div></label>
      ${formActions()}
    `;
  } else {
    const classOptions = [`<option value="">Chưa phân lớp</option>`].concat(
      (state.management.classrooms || []).map(classroom =>
        `<option value="${classroom.id}" ${classroom.id === item.class_id ? "selected" : ""}>${escapeHtml(classroom.name)}</option>`
      )
    ).join("");
    $("editorEyebrow").textContent = item.id ? "Cập nhật hồ sơ" : "Hồ sơ mới";
    $("editorTitle").textContent = item.id ? item.name : "Thêm người học";
    form.innerHTML = `
      <input type="hidden" name="entity" value="person"><input type="hidden" name="id" value="${escapeHtml(item.id || "")}">
      <div class="form-grid">
        ${field("name", "Họ và tên", item.name, "Nguyễn Văn A", true, "wide")}
        ${field("student_code", "Mã người học", item.student_code, "Mã sinh viên", true)}
        <label><span>Lớp học</span><select name="class_id">${classOptions}</select></label>
        ${field("desk_num", "Bàn", item.desk_num || "", "Để trống nếu chưa xếp", false, "", "number")}
        ${field("slot_num", "Chỗ", item.slot_num || 1, "", false, "", "number")}
      </div>
      <label class="face-upload">
        <span>Ảnh khuôn mặt</span>
        <div class="face-preview">
          ${item.has_face
            ? `<img id="facePreviewImage" src="/api/students/face?id=${encodeURIComponent(item.id)}" alt="Ảnh khuôn mặt hiện tại">`
            : `<div id="facePreviewEmpty">Chọn ảnh rõ mặt, nhìn thẳng</div>`}
        </div>
        <input id="faceInput" name="face_file" type="file" accept="image/jpeg,image/png,image/webp">
      </label>
      ${item.has_face ? `<label class="check-option"><input name="remove_face" type="checkbox" value="1">Xóa ảnh khuôn mặt hiện tại</label>` : ""}
      <p class="form-note">Ảnh sẽ được crop khuôn mặt lớn nhất và chuẩn hóa trước khi dùng cho nhận diện. Tối đa 8 MB.</p>
      ${formActions()}
    `;
  }
  const faceInput = $("faceInput");
  if (faceInput) faceInput.addEventListener("change", previewFaceFile);
  form.querySelector("input:not([type=hidden]), select")?.focus();
}

function field(name, label, value = "", placeholder = "", required = false, className = "", type = "text") {
  return `<label class="${className}"><span>${label}</span><input name="${name}" type="${type}" ${required ? "required" : ""} value="${escapeHtml(value || "")}" placeholder="${escapeHtml(placeholder)}"></label>`;
}

function formActions() {
  return `<div class="form-actions"><button type="button" class="button ghost" data-action="close-editor">Hủy</button><button type="submit" class="button primary">Lưu thay đổi</button></div>`;
}

function closeEditor() {
  $("editorPanel").classList.remove("open");
  $("editorPanel").setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

async function submitEditor(event) {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const entity = data.get("entity");
  let endpoint;
  let payload;
  if (entity === "class") {
    endpoint = "/api/classrooms";
    payload = { id: data.get("id"), name: data.get("name"), num_desks: Number(data.get("num_desks")) };
  } else if (entity === "camera") {
    endpoint = "/api/cameras";
    payload = Object.fromEntries(data.entries());
    payload.class_ids = data.getAll("class_ids");
    payload.port = Number(payload.port || 554);
    payload.isActive = true;
  } else {
    endpoint = "/api/students";
    payload = Object.fromEntries(data.entries());
    payload.desk_num = Number(payload.desk_num || 0);
    payload.slot_num = Number(payload.slot_num || 1);
    const faceFile = data.get("face_file");
    delete payload.face_file;
    payload.remove_face = data.get("remove_face") === "1";
    if (faceFile instanceof File && faceFile.size) {
      if (faceFile.size > 8 * 1024 * 1024) throw new Error("Ảnh khuôn mặt phải nhỏ hơn 8 MB");
      payload.face_data = await fileToDataUrl(faceFile);
      payload.remove_face = false;
    }
  }
  await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
  closeEditor();
  await loadManagement();
  await loadBootstrap();
  toast("Đã lưu thay đổi");
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Không đọc được file ảnh"));
    reader.readAsDataURL(file);
  });
}

function previewFaceFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    event.target.value = "";
    toast("Vui lòng chọn file ảnh", true);
    return;
  }
  const preview = event.target.closest(".face-upload").querySelector(".face-preview");
  const url = URL.createObjectURL(file);
  preview.innerHTML = `<img src="${url}" alt="Ảnh khuôn mặt vừa chọn">`;
}

async function openMonitor(classId = "", cameraId = "") {
  state.classId = classId || state.classId;
  state.cameraId = cameraId || "";
  showView("monitor");
  await loadBootstrap();
}

function handleManagementAction(action, id) {
  if (action === "new-class") return openEditor("class");
  if (action === "new-camera") return openEditor("camera");
  if (action === "new-person") return openEditor("person");
  if (action === "edit-class") return openEditor("class", state.management.classrooms.find(item => item.id === id));
  if (action === "edit-camera") return openEditor("camera", state.management.cameras.find(item => item.id === id));
  if (action === "edit-person") return openEditor("person", state.management.students.find(item => item.id === id));
  if (action === "monitor-class") return openMonitor(id);
  if (action === "monitor-camera") {
    const camera = state.management.cameras.find(item => item.id === id);
    return openMonitor(camera?.class_ids?.[0] || "", id);
  }
}

async function loadBootstrap() {
  const query = new URLSearchParams();
  if (state.classId) query.set("class_id", state.classId);
  if (state.cameraId) query.set("camera_id", state.cameraId);
  state.data = await api(`/api/bootstrap?${query}`);
  state.classId = state.data.class_id || "";
  state.cameraId = state.data.camera_id || "";
  state.seats = state.data.seats || [];
  state.students = state.data.students || {};
  renderSelectors();
  renderMapping();
  renderDesks();
  renderConnection();
  if (state.data.settings?.live_preview_on) {
    await startStream();
  } else {
    const stream = $("stream");
    stream.removeAttribute("src");
    $("emptyStream").textContent = "AI đang xử lý video record nền";
    $("emptyStream").classList.remove("hidden");
  }
  drawOverlay();
  await ensureAutoAI();
}

function renderConnection() {
  $("redisDot").className = `dot ${state.data.redis_ok ? "ok" : "bad"}`;
  $("redisText").textContent = state.data.redis_ok ? "Redis đã kết nối" : "Redis mất kết nối";
}

function compatibleCameras() {
  return (state.data.cameras || []).filter(
    camera => !state.classId || (camera.class_ids || []).includes(state.classId)
  );
}

function renderSelectors() {
  const classSelect = $("classSelect");
  classSelect.innerHTML = "";
  for (const classroom of state.data.classrooms || []) {
    classSelect.append(option(classroom.id, classroom.name, classroom.id === state.classId));
  }
  const cameraSelect = $("cameraSelect");
  cameraSelect.innerHTML = "";
  for (const camera of compatibleCameras()) {
    cameraSelect.append(option(
      camera.id,
      `${camera.name}${camera.stream_ready ? "" : " — chưa có stream"}`,
      camera.id === state.cameraId
    ));
  }
  const selectedCamera = compatibleCameras().find(camera => camera.id === state.cameraId);
  $("cameraTitle").textContent = selectedCamera?.name || "Chưa có camera";
}

function renderMapping() {
  const select = $("deskSelect");
  const previous = select.value;
  select.innerHTML = "";
  for (const seat of state.seats) {
    const desk = Number(seat.desk_num);
    select.append(option(
      String(desk),
      `Bàn ${desk}${seat.zone && Object.keys(seat.zone).length ? " — đã khoanh" : ""}`,
      previous ? String(desk) === previous : desk === 1
    ));
  }
}

function resultMap() {
  const map = new Map();
  for (const result of state.ai?.results || []) map.set(Number(result.desk_num), result);
  return map;
}

function renderDesks() {
  const box = $("deskList");
  const results = resultMap();
  box.innerHTML = "";
  let totalPeople = 0;

  for (const seat of state.seats) {
    const desk = Number(seat.desk_num);
    const result = results.get(desk);
    totalPeople += result?.present_count || 0;
    const details = document.createElement("details");
    details.className = "desk";
    details.open = desk <= 2;
    const slots = seat.slots || [];
    details.innerHTML = `
      <summary>
        <div class="desk-line">
          <span class="desk-title">Bàn ${desk}</span>
          <span class="desk-count">${result ? `${result.present_count}/${slots.length} người` : "Đang chờ AI"}</span>
        </div>
      </summary>
      <div class="slots"></div>
    `;
    const slotBox = details.querySelector(".slots");
    const slotResults = new Map((result?.slot_results || []).map(item => [Number(item.slot_num), item]));
    for (const slot of slots) {
      const student = state.students[slot.student_id] || {};
      const detection = slotResults.get(Number(slot.slot_num));
      const status = detection?.match_status || "";
      const statusClass = status === "correct" ? "correct" :
        ["wrong", "unassigned"].includes(status) ? "wrong" :
        detection?.present ? "present" : "";
      let statusText = detection?.match_text ||
        (detection?.present ? "Có người" : result ? "Trống / vắng" : "Chờ AI");
      if (detection?.gaze_alert) {
        const yaw = Number(detection.gaze_yaw_deg);
        const yawText = Number.isFinite(yaw) ? ` (yaw=${yaw.toFixed(0)}°)` : "";
        statusText += `  🚨 KHÔNG TẬP TRUNG${yawText}`;
      }
      if (detection?.absent) {
        const min = Number(detection.missing_for ?? 0) / 60;
        statusText += `  🛑 VẮNG MẶT (${min.toFixed(1)} phút)`;
      }
      const row = document.createElement("div");
      row.className = "slot";
      row.innerHTML = `
        <span>Chỗ ${slot.slot_num}</span>
        <span class="slot-name">${student.student_code ? `${student.student_code} · ` : ""}${student.name || "Chưa gán"}</span>
        <span class="slot-state ${statusClass}">${statusText}</span>
      `;
      slotBox.append(row);
    }
    box.append(details);
  }

  $("summaryTitle").textContent = state.aiActive
    ? `${totalPeople} người trong vùng`
    : `${state.seats.length} bàn đang theo dõi`;
}

async function startStream() {
  const stream = $("stream");
  $("emptyStream").classList.remove("hidden");
  if (!state.cameraId) return;
  try {
    const data = await api(`/api/stream-url?camera_id=${encodeURIComponent(state.cameraId)}&fps=25`);
    stream.onload = () => {
      $("emptyStream").classList.add("hidden");
      resizeOverlay();
    };
    stream.onerror = () => {
      $("emptyStream").textContent = "Không nhận được video record";
      $("emptyStream").classList.remove("hidden");
    };
    stream.src = `${data.url}&_=${Date.now()}`;
  } catch (error) {
    $("emptyStream").textContent = error.message;
    toast(error.message, true);
  }
}

function resizeOverlay() {
  const viewport = $("viewport");
  const canvas = $("overlay");
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(viewport.clientWidth * ratio);
  canvas.height = Math.round(viewport.clientHeight * ratio);
  canvas.style.width = `${viewport.clientWidth}px`;
  canvas.style.height = `${viewport.clientHeight}px`;
  drawOverlay();
}

function canvasContext() {
  const canvas = $("overlay");
  const ratio = window.devicePixelRatio || 1;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: canvas.clientWidth, height: canvas.clientHeight };
}

function imageRect(width, height) {
  const image = $("stream");
  const naturalWidth = image.naturalWidth || 16;
  const naturalHeight = image.naturalHeight || 9;
  const scale = Math.min(width / naturalWidth, height / naturalHeight);
  const w = naturalWidth * scale;
  const h = naturalHeight * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}

function drawOverlay() {
  const { ctx, width, height } = canvasContext();
  ctx.clearRect(0, 0, width, height);
  const rect = imageRect(width, height);
  ctx.font = "700 12px Geist, sans-serif";
  ctx.lineWidth = 2;

  for (const seat of state.seats) {
    const zone = seat.zone || {};
    if (!Object.keys(zone).length) continue;
    const desk = Number(seat.desk_num);
    const color = deskColor(desk);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    if (zone.type === "oriented") {
      const cx = rect.x + Number(zone.cx) * rect.w;
      const cy = rect.y + Number(zone.cy) * rect.h;
      const zw = Number(zone.w) * rect.w;
      const zh = Number(zone.h) * rect.h;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(Number(zone.angle || 0) * Math.PI / 180);
      ctx.strokeRect(-zw / 2, -zh / 2, zw, zh);
      ctx.restore();
      drawDeskTag(ctx, `BÀN ${desk}`, cx - zw / 2 + 4, cy - zh / 2 - 23, color);
    } else {
      const x = rect.x + Number(zone.x) * rect.w;
      const y = rect.y + Number(zone.y) * rect.h;
      const w = Number(zone.w) * rect.w;
      const h = Number(zone.h) * rect.h;
      ctx.strokeRect(x, y, w, h);
      drawDeskTag(ctx, `BÀN ${desk}`, x + 4, Math.max(0, y - 23), color);
    }
  }

  // Person boxes are only painted on the exact frame used by inference.
  // Live recorded video remains smooth but intentionally has no stale boxes.
  if (state.snapshotMode) drawDetections(ctx, rect);

  if (state.draft) {
    const x = rect.x + state.draft.x * rect.w;
    const y = rect.y + state.draft.y * rect.h;
    const w = state.draft.w * rect.w;
    const h = state.draft.h * rect.h;
    ctx.fillStyle = "rgba(184, 255, 53, 0.14)";
    ctx.strokeStyle = "#b8ff35";
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }
}

function drawDeskTag(ctx, text, x, y, color) {
  const width = ctx.measureText(text).width + 12;
  ctx.fillStyle = color;
  ctx.fillRect(x, y, width, 19);
  ctx.fillStyle = textColorFor(color);
  ctx.fillText(text, x + 6, y + 14);
}

function drawDetections(ctx, rect) {
  const firstResult = (state.ai?.results || [])[0] || {};
  const sourceWidth = Math.max(1, Number(firstResult.source_width || 1920));
  const sourceHeight = Math.max(1, Number(firstResult.source_height || 1080));

  for (const deskResult of state.ai?.results || []) {
    const deskSourceWidth = Math.max(1, Number(deskResult.source_width || sourceWidth));
    const deskSourceHeight = Math.max(1, Number(deskResult.source_height || sourceHeight));
    const detections = [
      ...(deskResult.slot_results || []),
      ...(deskResult.extra_results || []),
    ];
    for (const detection of detections) {
      const bbox = detection.person_bbox;
      if (!bbox || bbox.length < 4) continue;
      if (detection.match_status !== "correct") continue;
      const [bx, by, bw, bh] = bbox.map(Number);
      const x = rect.x + bx / deskSourceWidth * rect.w;
      const y = rect.y + by / deskSourceHeight * rect.h;
      const w = bw / deskSourceWidth * rect.w;
      const h = bh / deskSourceHeight * rect.h;
      const status = detection.match_status || "";
      const desk = Number(deskResult.desk_num);
      const distracted = Boolean(detection.gaze_alert);
      const color = distracted ? "#ef4444" : deskColor(desk);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = distracted ? 4 : 3;
      ctx.strokeRect(x, y, w, h);
      const name = detection.recognized_name || "NGƯỜI";
      const slot = Number(detection.slot_num);
      const statusMark = status === "correct" ? "✓" :
        ["wrong", "unassigned"].includes(status) ? "!" : "";
      const label = slot >= 0
        ? `B${desk}.${slot}${statusMark} · ${name}`
        : `B${desk}.?${statusMark} · ${name}`;
      const labelWidth = ctx.measureText(label).width + 10;
      const labelY = Math.max(0, y - 20);
      ctx.fillRect(x, labelY, labelWidth, 20);
      ctx.fillStyle = textColorFor(color);
      ctx.fillText(label, x + 5, labelY + 14);

      if (distracted) {
        const yaw = Number(detection.gaze_yaw_deg);
        const yawText = Number.isFinite(yaw) ? ` (yaw=${yaw.toFixed(0)}°)` : "";
        const alertLabel = `⚠ KHÔNG TẬP TRUNG${yawText}`;
        const alertWidth = ctx.measureText(alertLabel).width + 10;
        const alertY = Math.min(y + h + 2, rect.y + rect.h - 20);
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(x, alertY, alertWidth, 20);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(alertLabel, x + 5, alertY + 14);
      }
    }
  }
}

function normalizedPoint(event) {
  const canvas = $("overlay");
  const bounds = canvas.getBoundingClientRect();
  const rect = imageRect(bounds.width, bounds.height);
  return {
    x: Math.max(0, Math.min(1, (event.clientX - bounds.left - rect.x) / rect.w)),
    y: Math.max(0, Math.min(1, (event.clientY - bounds.top - rect.y) / rect.h)),
  };
}

function installDrawing() {
  const canvas = $("overlay");
  canvas.addEventListener("pointerdown", event => {
    if (!state.drawing) return;
    state.dragStart = normalizedPoint(event);
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!state.drawing || !state.dragStart) return;
    const point = normalizedPoint(event);
    state.draft = {
      type: "normal",
      x: Math.min(state.dragStart.x, point.x),
      y: Math.min(state.dragStart.y, point.y),
      w: Math.abs(point.x - state.dragStart.x),
      h: Math.abs(point.y - state.dragStart.y),
    };
    $("saveButton").disabled = state.draft.w < 0.005 || state.draft.h < 0.005;
    drawOverlay();
  });
  canvas.addEventListener("pointerup", event => {
    if (!state.drawing) return;
    canvas.releasePointerCapture(event.pointerId);
    state.dragStart = null;
  });
}

async function pollAI() {
  try {
    state.ai = await api("/api/ai/status");
    state.aiActive = Boolean(
      state.ai.active || state.ai.requested || state.ai.loading || state.ai.running
    );
    const detectionCount = (state.ai.results || []).reduce(
      (total, result) => total +
        (result.slot_results || []).filter(item => item.person_bbox).length +
        (result.extra_results || []).filter(item => item.person_bbox).length,
      0
    );
    const yoloCount = (state.ai.results?.[0]?.all_person_boxes || []).length;
    $("aiToggle").textContent = state.ai?.loading
      ? "Đang khởi động AI"
      : state.aiActive ? "Dừng AI" : "Bật AI";
    $("aiToggle").className = `button ${state.aiActive ? "danger" : "primary"}`;
    $("aiStatus").textContent = state.ai.error || (
      state.ai.sequence
        ? `${state.ai.status} · frame ${state.ai.frame_count} · YOLO ${yoloCount} · trong vùng ${detectionCount}`
        : state.ai.status
    );
    $("latency").textContent = state.ai.inference_ms
      ? `AI ${state.ai.inference_ms} ms · frame ${state.ai.frame_count ?? "—"}`
      : state.ai.status;
    $("lastUpdate").textContent = state.ai.updated_at
      ? `Cập nhật ${new Date(state.ai.updated_at * 1000).toLocaleTimeString("vi-VN")}`
      : "Chưa có kết quả";
    if (state.snapshotMode && state.ai.sequence) {
      const stream = $("stream");
      const next = `/api/ai/snapshot.jpg?seq=${state.ai.sequence}`;
      if (!stream.src.endsWith(next)) stream.src = next;
      state.waitingForSnapshot = false;
      $("emptyStream").classList.add("hidden");
    } else if (state.snapshotMode && state.waitingForSnapshot) {
      $("emptyStream").textContent = "Đang chờ AI xử lý video đầu tiên";
      $("emptyStream").classList.remove("hidden");
    }
    renderDesks();
    drawOverlay();
  } catch (error) {
    $("aiStatus").textContent = error.message;
  }
}

async function toggleAI() {
  if (state.aiChanging) return;
  state.aiChanging = true;
  $("aiToggle").disabled = true;
  try {
    if (state.aiActive) {
      await api("/api/ai/stop", { method: "POST", body: "{}" });
      state.snapshotMode = false;
      state.waitingForSnapshot = false;
      updateViewLabels();
      if (state.data?.settings?.live_preview_on) {
        await startStream();
      }
    } else {
      const response = await startAI();
      state.ai = response.state;
      state.aiActive = Boolean(
        state.ai?.active || state.ai?.requested || state.ai?.loading || state.ai?.running
      );
      state.snapshotMode = Boolean(state.data?.settings?.snapshot_on);
      state.waitingForSnapshot = state.snapshotMode;
      updateViewLabels();
    }
    await pollAI();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.aiChanging = false;
    $("aiToggle").disabled = false;
  }
}

async function startAI() {
  return api("/api/ai/start", {
    method: "POST",
    body: JSON.stringify({
      class_id: state.classId,
      camera_id: state.cameraId,
      mode: $("modeSelect").value,
    }),
  });
}

async function ensureAutoAI() {
  if (!state.data?.settings?.ai_auto_start) return;
  if (!state.classId || !state.cameraId || state.aiActive || state.aiChanging) return;
  state.aiChanging = true;
  $("aiToggle").disabled = true;
  try {
    const response = await startAI();
    state.ai = response.state;
    state.aiActive = Boolean(
      state.ai?.active || state.ai?.requested || state.ai?.loading || state.ai?.running
    );
    state.snapshotMode = false;
    state.waitingForSnapshot = false;
    updateViewLabels();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.aiChanging = false;
    $("aiToggle").disabled = false;
  }
}

async function toggleView() {
  if (!state.data?.settings?.snapshot_on) {
    toast("Snapshot UI đang tắt để đo hiệu năng AI", true);
    return;
  }
  state.snapshotMode = !state.snapshotMode;
  if (state.snapshotMode) {
    if (!state.ai?.sequence) {
      state.snapshotMode = false;
      toast("Chưa có kết quả AI để đối chiếu", true);
      updateViewLabels();
      return;
    }
    $("stream").src = `/api/ai/snapshot.jpg?seq=${state.ai.sequence}`;
  } else {
    if (state.data?.settings?.live_preview_on) {
      await startStream();
    }
  }
  updateViewLabels();
}

function updateViewLabels() {
  $("viewToggle").textContent = state.snapshotMode ? "Quay lại Live 25 FPS" : "Khung AI đồng bộ";
  $("viewMode").textContent = state.snapshotMode ? "AI SYNC FRAME" : "LIVE 25 FPS";
  $("eyebrow").textContent = state.snapshotMode
    ? "Frame đúng thời điểm suy luận"
    : "Live từ video record · không ghép hộp AI cũ";
}

async function saveZone() {
  if (!state.draft) return;
  const restartAI = state.aiActive;
  try {
    if (restartAI) await api("/api/ai/stop", { method: "POST", body: "{}" });
    await api("/api/zones", {
      method: "POST",
      body: JSON.stringify({
        class_id: state.classId,
        camera_id: state.cameraId,
        desk_num: Number($("deskSelect").value),
        zone: state.draft,
      }),
    });
    cancelDrawing();
    await loadBootstrap();
    if (restartAI) await startAI();
    toast("Đã lưu vùng bàn");
  } catch (error) {
    toast(error.message, true);
  }
}

async function deleteZone() {
  const restartAI = state.aiActive;
  try {
    if (restartAI) await api("/api/ai/stop", { method: "POST", body: "{}" });
    await api("/api/zones/delete", {
      method: "POST",
      body: JSON.stringify({
        class_id: state.classId,
        camera_id: state.cameraId,
        desk_num: Number($("deskSelect").value),
      }),
    });
    await loadBootstrap();
    if (restartAI) await startAI();
    toast("Đã xóa vùng bàn");
  } catch (error) {
    toast(error.message, true);
  }
}

function beginDrawing() {
  if (state.snapshotMode) {
    toast("Hãy quay lại Live 25 FPS trước khi vẽ", true);
    return;
  }
  state.drawing = true;
  state.draft = null;
  $("viewport").classList.add("drawing");
  $("saveButton").disabled = true;
  toast("Kéo chuột trên video để khoanh vùng");
}

function cancelDrawing() {
  state.drawing = false;
  state.dragStart = null;
  state.draft = null;
  $("viewport").classList.remove("drawing");
  $("saveButton").disabled = true;
  drawOverlay();
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach(button => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });
  document.addEventListener("click", event => {
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    if (action === "close-editor") return closeEditor();
    handleManagementAction(action, target.dataset.id);
  });
  $("editorForm").addEventListener("submit", event => {
    submitEditor(event).catch(error => toast(error.message, true));
  });
  $("peopleClassFilter").addEventListener("change", renderPeopleManagement);
  $("peopleSearch").addEventListener("input", renderPeopleManagement);
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeEditor();
  });
  $("classSelect").addEventListener("change", async event => {
    if (state.aiActive) await api("/api/ai/stop", { method: "POST", body: "{}" });
    state.ai = null;
    state.aiActive = false;
    state.snapshotMode = false;
    updateViewLabels();
    state.classId = event.target.value;
    state.cameraId = "";
    await loadBootstrap();
  });
  $("cameraSelect").addEventListener("change", async event => {
    if (state.aiActive) await api("/api/ai/stop", { method: "POST", body: "{}" });
    state.ai = null;
    state.aiActive = false;
    state.snapshotMode = false;
    updateViewLabels();
    state.cameraId = event.target.value;
    await loadBootstrap();
  });
  $("aiToggle").addEventListener("click", toggleAI);
  $("viewToggle").addEventListener("click", toggleView);
  $("refreshButton").addEventListener("click", loadBootstrap);
  $("drawButton").addEventListener("click", beginDrawing);
  $("saveButton").addEventListener("click", saveZone);
  $("cancelButton").addEventListener("click", cancelDrawing);
  $("deleteButton").addEventListener("click", deleteZone);
  window.addEventListener("resize", resizeOverlay);
  installDrawing();
}

async function main() {
  bindEvents();
  try {
    await loadBootstrap();
    await loadManagement();
    showView(location.hash.replace("#", "") || "monitor", false);
    await pollAI();
    state.pollTimer = setInterval(pollAI, 2000);
  } catch (error) {
    toast(error.message, true);
    $("emptyStream").textContent = error.message;
  }
}

main();
