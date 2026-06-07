const API = '';
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

let currentDocId = null;
let currentPageIndex = 0;
let currentPageData = null;
let currentPages = [];
let currentElements = [];
let currentDocument = null;
let editingElementId = null;
let isEditOrderMode = false;
let originalOrder = [];
let currentPdfDoc = null;
let currentScale = 1.5;
let activeElementId = null;
let draggedElement = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function init() {
    setupRouter();
    setupDropZone();
    setupViewToggle();
    handleRoute();
}

function setupRouter() {
    window.addEventListener('hashchange', handleRoute);
}

function handleRoute() {
    const hash = window.location.hash || '#/home';
    const parts = hash.replace('#/', '').split('/');
    const route = parts[0];

    $$('.page').forEach(p => p.classList.remove('active'));
    $$('.nav-item').forEach(n => n.classList.remove('active'));

    if (route === 'home') {
        $('#page-home').classList.add('active');
        document.querySelector('.nav-item[data-route="home"]').classList.add('active');
        loadDocuments();
    } else if (route === 'detail' && parts[1]) {
        $('#page-detail').classList.add('active');
        document.querySelector('.nav-item[data-route="home"]').classList.add('active');
        loadDocumentDetail(parseInt(parts[1]));
    } else {
        navigateTo('home');
    }
}

function navigateTo(route) {
    window.location.hash = `#/${route}`;
}

function setupDropZone() {
    const zone = $('#drop-zone');
    const input = $('#file-input');

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.name.toLowerCase().endsWith('.pdf')) {
            uploadFile(file);
        } else {
            showUploadResult('仅支持 PDF 文件', true);
        }
    });

    input.addEventListener('change', () => {
        if (input.files[0]) {
            uploadFile(input.files[0]);
        }
    });
}

function setupViewToggle() {
    $$('.view-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            $$('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            $('.documents-list-view').classList.toggle('active', view === 'list');
            $('.documents-grid-view').classList.toggle('active', view === 'grid');
        });
    });
}

async function uploadFile(file) {
    const progressEl = $('#upload-progress');
    const resultEl = $('#upload-result');
    progressEl.classList.remove('hidden');
    resultEl.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', API + '/api/upload');

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progressEl.querySelector('.progress-fill').style.width = pct + '%';
                progressEl.querySelector('.progress-text').textContent = pct + '%';
            }
        };

        const result = await new Promise((resolve, reject) => {
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    try {
                        reject(JSON.parse(xhr.responseText));
                    } catch {
                        reject({ detail: xhr.statusText });
                    }
                }
            };
            xhr.onerror = () => reject({ detail: 'Network error' });
            xhr.send(formData);
        });

        showUploadResult(`上传成功！文档 ID: ${result.document_id}，共 ${result.page_count} 页`, false);
        loadDocuments();

        setTimeout(() => {
            startParsing(result.document_id);
        }, 500);
    } catch (err) {
        showUploadResult(err.detail || '上传失败', true);
    }
}

function showUploadResult(msg, isError) {
    const el = $('#upload-result');
    el.textContent = msg;
    el.className = 'result-msg ' + (isError ? 'error' : 'success');
    el.classList.remove('hidden');
}

async function loadDocuments() {
    try {
        const res = await fetch(API + '/api/documents');
        const data = await res.json();
        renderDocumentsList(data.documents);
        renderDocumentsGrid(data.documents);
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

function formatDateTime(isoStr) {
    if (!isoStr) return '-';
    const d = new Date(isoStr);
    return d.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getStatusText(status) {
    const map = {
        'uploaded': '已上传',
        'validated': '已验证',
        'pages_ready': '待解析',
        'processing': '处理中',
        'parsing_layout': '布局解析中',
        'parsing_content': '内容解析中',
        'completed': '已完成',
        'failed': '失败'
    };
    return map[status] || status;
}

function getResultText(status) {
    if (status === 'completed') return '<span class="parse-result success"><i class="fas fa-check-circle"></i> 解析成功</span>';
    if (status === 'failed') return '<span class="parse-result failed"><i class="fas fa-times-circle"></i> 解析失败</span>';
    if (status === 'processing' || status.startsWith('parsing')) return '<span class="parse-result processing"><i class="fas fa-spinner fa-spin"></i> 解析中</span>';
    return '<span class="parse-result"><i class="fas fa-clock"></i> 待解析</span>';
}

function renderDocumentsList(docs) {
    const container = $('#documents-list');
    if (!docs || docs.length === 0) {
        container.innerHTML = '<p class="empty-msg"><i class="fas fa-inbox"></i><br>暂无文档，请上传 PDF 文件开始使用</p>';
        return;
    }

    container.innerHTML = `
        <table class="doc-list-table">
            <thead>
                <tr>
                    <th>文件名</th>
                    <th>上传时间</th>
                    <th>解析时间</th>
                    <th>解析结果</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                ${docs.map(doc => `
                    <tr>
                        <td class="doc-name-cell" onclick="viewDocument(${doc.id})">
                            <i class="fas fa-file-pdf" style="color: var(--error); margin-right: 0.5rem;"></i>
                            ${escapeHtml(doc.original_filename)}
                        </td>
                        <td class="doc-time">${formatDateTime(doc.created_at)}</td>
                        <td class="doc-time">${doc.status === 'completed' ? formatDateTime(doc.updated_at) : '-'}</td>
                        <td>${getResultText(doc.status)}</td>
                        <td class="doc-actions">
                            ${doc.status !== 'processing' && !doc.status.startsWith('parsing') ? 
                                `<button class="btn btn-primary btn-sm" onclick="event.stopPropagation(); startParsing(${doc.id})">
                                    <i class="fas fa-play"></i> 解析
                                </button>` : ''}
                            ${doc.status === 'completed' ? 
                                `<button class="btn btn-warning btn-sm" onclick="event.stopPropagation(); reparseDocumentFromList(${doc.id})">
                                    <i class="fas fa-sync-alt"></i> 重解析
                                </button>` : ''}
                            <button class="btn btn-success btn-sm" onclick="event.stopPropagation(); viewDocument(${doc.id})">
                                <i class="fas fa-eye"></i> 查看
                            </button>
                            <button class="btn btn-danger btn-sm" onclick="event.stopPropagation(); deleteDocument(${doc.id})">
                                <i class="fas fa-trash"></i> 删除
                            </button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderDocumentsGrid(docs) {
    const container = $('#documents-grid');
    if (!docs || docs.length === 0) {
        container.innerHTML = '<p class="empty-msg"><i class="fas fa-inbox"></i><br>暂无文档</p>';
        return;
    }

    container.innerHTML = docs.map(doc => `
        <div class="book-card" onclick="viewDocument(${doc.id})">
            <div class="book-cover">
                ${doc.status === 'completed' ? 
                    `<img src="${API}/api/documents/${doc.id}/thumbnail" alt="${escapeHtml(doc.original_filename)}" onerror="this.outerHTML='<div class=\\'book-cover-placeholder\\'><i class=\\'fas fa-file-pdf\\'></i>待解析</div>'">` :
                    `<div class="book-cover-placeholder">
                        <i class="fas fa-file-pdf"></i>
                        ${doc.status === 'processing' || doc.status.startsWith('parsing') ? '解析中...' : '待解析'}
                    </div>`
                }
            </div>
            <div class="book-info">
                <div class="book-title" title="${escapeHtml(doc.original_filename)}">${escapeHtml(doc.original_filename)}</div>
                <div class="book-meta">
                    <span class="status-badge status-${doc.status}">${getStatusText(doc.status)}</span>
                    <span>${doc.page_count} 页</span>
                </div>
            </div>
        </div>
    `).join('');
}

async function startParsing(docId) {
    try {
        const res = await fetch(API + '/api/parse/' + docId, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to start parsing');
        alert('解析已开始，请稍候...');
        loadDocuments();
    } catch (e) {
        alert('启动解析失败: ' + e.message);
    }
}

async function reparseDocumentFromList(docId) {
    if (!confirm('确定要重新解析此文档吗？这将清除现有解析结果。')) return;
    try {
        const res = await fetch(API + '/api/reparse/' + docId, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to start reparsing');
        alert('重解析已开始，请稍候...');
        loadDocuments();
    } catch (e) {
        alert('启动重解析失败: ' + e.message);
    }
}

async function deleteDocument(docId) {
    if (!confirm('确定要删除此文档及其解析结果吗？')) return;

    try {
        await fetch(API + '/api/documents/' + docId, { method: 'DELETE' });
        loadDocuments();
    } catch (e) {
        alert('删除失败: ' + e.message);
    }
}

function viewDocument(docId) {
    navigateTo(`detail/${docId}`);
}

async function loadDocumentDetail(docId) {
    currentDocId = docId;
    currentPageIndex = 0;
    activeElementId = null;
    isEditOrderMode = false;

    try {
        const [docRes, pagesRes, resultsRes] = await Promise.all([
            fetch(API + '/api/documents').then(r => r.json()),
            fetch(API + '/api/status/' + docId).then(r => r.json()),
            fetch(API + '/api/results/' + docId).then(r => r.json()).catch(() => null)
        ]);

        currentDocument = docRes.documents.find(d => d.id === docId);
        currentPages = pagesRes.pages;

        if (!currentDocument) {
            alert('文档不存在');
            navigateTo('home');
            return;
        }

        $('#detail-title').textContent = currentDocument.original_filename;
        const statusEl = $('#detail-status');
        statusEl.textContent = getStatusText(currentDocument.status);
        statusEl.className = 'status-badge status-' + currentDocument.status;

        renderThumbnails();

        if (currentPages.length > 0) {
            loadPage(0);
        }
    } catch (e) {
        console.error('Failed to load document detail:', e);
        alert('加载文档详情失败: ' + e.message);
    }
}

function renderThumbnails() {
    const container = $('#thumbs-list');
    container.innerHTML = currentPages.map((page, idx) => {
        const imgSrc = page.jpg_path ? `${API}/api/file/${encodeURIComponent(page.jpg_path)}` : null;

        return `
            <div class="thumb-item ${idx === currentPageIndex ? 'active' : ''}" data-index="${idx}" onclick="loadPage(${idx})">
                ${imgSrc ? 
                    `<img src="${imgSrc}" alt="第 ${page.page_number} 页" onerror="this.outerHTML='<div class=\\'thumb-placeholder\\'>第 ${page.page_number} 页</div>'">` :
                    `<div class="thumb-placeholder">第 ${page.page_number} 页</div>`
                }
                <div class="thumb-page-num">P. ${page.page_number}</div>
            </div>
        `;
    }).join('');
}

function getPageDataByNumber(pageNum) {
    return currentPages.find(p => p.page_number === pageNum) || null;
}

async function loadPage(index) {
    currentPageIndex = index;
    const page = currentPages[index];

    $$('.thumb-item').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });

    $('#pdf-page-info').textContent = `第 ${page.page_number} 页`;

    try {
        const res = await fetch(API + '/api/pages/' + page.id + '/elements');
        const data = await res.json();
        currentElements = data.elements;
        currentPageData = page;

        await renderPdfPage(page);
        renderElements();
        clearAnnotations();
    } catch (e) {
        console.error('Failed to load page:', e);
        currentElements = [];
        renderElements();
    }
}

async function renderPdfPage(page) {
    const canvas = $('#pdf-canvas');
    const ctx = canvas.getContext('2d');
    const annotationLayer = $('#annotation-layer');

    try {
        if (page.single_pdf_path) {
            const pdfUrl = `${API}/api/pages/${page.id}/pdf`;
            const loadingTask = pdfjsLib.getDocument(pdfUrl);
            const pdfDoc = await loadingTask.promise;
            const pdfPage = await pdfDoc.getPage(1);

            const viewport = pdfPage.getViewport({ scale: currentScale });
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            annotationLayer.style.width = viewport.width + 'px';
            annotationLayer.style.height = viewport.height + 'px';

            await pdfPage.render({
                canvasContext: ctx,
                viewport: viewport
            }).promise;

            clearAnnotations();
            renderAnnotations(currentElements, viewport.width, viewport.height);
        } else if (page.jpg_path) {
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width * currentScale;
                canvas.height = img.height * currentScale;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                annotationLayer.style.width = canvas.width + 'px';
                annotationLayer.style.height = canvas.height + 'px';

                clearAnnotations();
                renderAnnotations(currentElements, canvas.width, canvas.height);
            };
            img.src = `${API}/api/file/${encodeURIComponent(page.jpg_path)}`;
        } else {
            canvas.width = 600;
            canvas.height = 800;
            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, 600, 800);
            ctx.fillStyle = '#64748b';
            ctx.font = '16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('暂无页面预览', 300, 400);
        }
    } catch (e) {
        console.error('Failed to render PDF:', e);
        canvas.width = 600;
        canvas.height = 800;
        ctx.fillStyle = '#fef2f2';
        ctx.fillRect(0, 0, 600, 800);
        ctx.fillStyle = '#ef4444';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('页面渲染失败: ' + e.message, 300, 400);
    }
}

function clearAnnotations() {
    $('#annotation-layer').innerHTML = '';
}

function renderAnnotations(elements, canvasWidth, canvasHeight) {
    const layer = $('#annotation-layer');
    if (!currentPageData) return;

    const page = currentPageData;
    const pageWidth = page.width || canvasWidth;
    const pageHeight = page.height || canvasHeight;

    elements.forEach(elem => {

        const scaleX = canvasWidth / pageWidth;
        const scaleY = canvasHeight / pageHeight;

        const x = elem.bbox_x0 * scaleX;
        const y = elem.bbox_y0 * scaleY;
        const w = (elem.bbox_x1 - elem.bbox_x0) * scaleX;
        const h = (elem.bbox_y1 - elem.bbox_y0) * scaleY;

        const box = document.createElement('div');
        box.className = `annotation-box type-${elem.element_type.toLowerCase()} ${activeElementId === elem.id ? 'active' : ''}`;
        box.dataset.type = elem.element_type;
        box.dataset.elementId = elem.id;
        box.style.left = x + 'px';
        box.style.top = y + 'px';
        box.style.width = w + 'px';
        box.style.height = h + 'px';

        box.addEventListener('click', () => {
            highlightElement(elem.id);
        });

        layer.appendChild(box);
    });
}

function highlightElement(elementId) {
    activeElementId = elementId;

    $$('.annotation-box').forEach(box => {
        box.classList.toggle('active', parseInt(box.dataset.elementId) === elementId);
    });

    $$('.element-card').forEach(card => {
        card.classList.toggle('active', parseInt(card.dataset.elementId) === elementId);
    });

    const card = document.querySelector(`.element-card[data-element-id="${elementId}"]`);
    if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function renderElements() {
    const container = $('#elements-list');
    container.classList.toggle('edit-order-mode', isEditOrderMode);

    if (!currentElements || currentElements.length === 0) {
        container.innerHTML = '<p class="empty-msg">当前页面暂无解析元素</p>';
        return;
    }

    const sorted = [...currentElements].sort((a, b) => a.reading_order - b.reading_order);

    container.innerHTML = sorted.map((elem, idx) => {
        const type = elem.element_type.toLowerCase();
        const isImage = elem.content_format === 'image_path';
        let contentHtml = '';

        if (isImage && elem.content) {
            contentHtml = `<img src="${API}/api/file/${encodeURIComponent(elem.content)}" alt="图片">`;
        } else if (elem.content_format === 'latex') {
            contentHtml = `<code>${escapeHtml(elem.content || '(空)')}</code>`;
        } else {
            contentHtml = renderMarkdownSimple(elem.content || '(空)');
        }

        return `
            <div class="element-card ${activeElementId === elem.id ? 'active' : ''}" 
                 data-element-id="${elem.id}"
                 data-order="${elem.reading_order}"
                 draggable="${isEditOrderMode}"
                 ondragstart="handleDragStart(event, ${elem.id})"
                 ondragend="handleDragEnd(event)"
                 ondragover="handleDragOver(event)"
                 ondragleave="handleDragLeave(event)"
                 ondrop="handleDrop(event, ${elem.id})">
                <div class="element-header" onclick="highlightElement(${elem.id})">
                    <span class="drag-handle" onclick="event.stopPropagation()">
                        <i class="fas fa-grip-vertical"></i>
                    </span>
                    <span class="element-type ${type}">${elem.element_type}</span>
                    <span class="element-order">#${elem.reading_order}</span>
                    <span class="element-confidence">${(elem.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="element-content markdown" onclick="highlightElement(${elem.id})">
                    ${contentHtml}
                </div>
                <div class="element-footer">
                    <button class="btn btn-outline btn-sm edit-btn" onclick="event.stopPropagation(); openEditModal(${elem.id})">
                        <i class="fas fa-edit"></i> 编辑
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function renderMarkdownSimple(text) {
    if (!text) return '';
    
    let html = escapeHtml(text);
    
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');
    
    html = html.replace(/^\|(.+)\|$/gm, (match) => {
        const cells = match.split('|').filter(c => c.trim());
        if (cells.every(c => /^[-:]+$/.test(c.trim()))) return '';
        return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
    });
    
    if (html.includes('<tr>')) {
        html = '<table>' + html.replace(/(<tr>.*?<\/tr>)/gs, '$1') + '</table>';
    }
    
    html = html.replace(/^- (.*$)/gm, '<li>$1</li>');
    html = html.replace(/^(\d+)\. (.*$)/gm, '<li>$2</li>');
    
    html = html.replace(/(<li>.*?<\/li>)(\n<li>)/gs, '$1$2');
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    
    html = html.replace(/<p><h(\d)>/g, '<h$1>');
    html = html.replace(/<\/h(\d)><\/p>/g, '</h$1>');
    html = html.replace(/<p><table>/g, '<table>');
    html = html.replace(/<\/table><\/p>/g, '</table>');
    html = html.replace(/<p><ul>/g, '<ul>');
    html = html.replace(/<\/ul><\/p>/g, '</ul>');
    html = html.replace(/<p><\/p>/g, '');
    
    return html;
}

function openEditModal(elementId) {
    const elem = currentElements.find(e => e.id === elementId);
    if (!elem) return;

    editingElementId = elementId;
    $('#edit-type').value = elem.element_type.toLowerCase();
    $('#edit-content').value = elem.content || '';
    $('#edit-modal').classList.remove('hidden');
}

function closeEditModal() {
    $('#edit-modal').classList.add('hidden');
    editingElementId = null;
}

async function saveElementEdit() {
    if (!editingElementId) return;

    const newType = $('#edit-type').value;
    const newContent = $('#edit-content').value;

    try {
        const res = await fetch(API + '/api/elements/' + editingElementId, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                element_type: newType,
                content: newContent
            })
        });

        if (!res.ok) throw new Error('Failed to update element');

        const updated = await res.json();
        const idx = currentElements.findIndex(e => e.id === editingElementId);
        if (idx !== -1) {
            currentElements[idx] = updated;
        }

        renderElements();
        
        if (currentPageData) {
            const canvas = $('#pdf-canvas');
            clearAnnotations();
            renderAnnotations(currentElements, canvas.width, canvas.height);
        }

        closeEditModal();
    } catch (e) {
        alert('保存失败: ' + e.message);
    }
}

function toggleEditOrder() {
    isEditOrderMode = true;
    originalOrder = currentElements.map(e => ({ id: e.id, order: e.reading_order }));
    $('#edit-order-btn').classList.add('hidden');
    $('#save-order-btn').classList.remove('hidden');
    $('#cancel-order-btn').classList.remove('hidden');
    renderElements();
}

function cancelOrder() {
    isEditOrderMode = false;
    currentElements.forEach(elem => {
        const orig = originalOrder.find(o => o.id === elem.id);
        if (orig) elem.reading_order = orig.order;
    });
    originalOrder = [];
    $('#edit-order-btn').classList.remove('hidden');
    $('#save-order-btn').classList.add('hidden');
    $('#cancel-order-btn').classList.add('hidden');
    renderElements();
}

async function saveOrder() {
    if (!currentPageData) return;

    const sorted = [...currentElements].sort((a, b) => a.reading_order - b.reading_order);
    const elementOrder = sorted.map(e => e.id);

    try {
        const res = await fetch(API + '/api/pages/' + currentPageData.id + '/elements/reorder', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ element_order })
        });

        if (!res.ok) throw new Error('Failed to reorder elements');

        sorted.forEach((elem, idx) => {
            elem.reading_order = idx;
        });

        isEditOrderMode = false;
        originalOrder = [];
        $('#edit-order-btn').classList.remove('hidden');
        $('#save-order-btn').classList.add('hidden');
        $('#cancel-order-btn').classList.add('hidden');
        renderElements();

        alert('排序已保存');
    } catch (e) {
        alert('保存排序失败: ' + e.message);
    }
}

function handleDragStart(e, elementId) {
    if (!isEditOrderMode) return;
    draggedElement = elementId;
    e.target.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragEnd(e) {
    if (!isEditOrderMode) return;
    e.target.classList.remove('dragging');
    document.querySelectorAll('.element-card').forEach(card => {
        card.classList.remove('drag-over');
    });
    draggedElement = null;
}

function handleDragOver(e) {
    if (!isEditOrderMode || !draggedElement) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    if (!isEditOrderMode) return;
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e, targetId) {
    if (!isEditOrderMode || !draggedElement || draggedElement === targetId) return;
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    const draggedElem = currentElements.find(e => e.id === draggedElement);
    const targetElem = currentElements.find(e => e.id === targetId);

    if (draggedElem && targetElem) {
        const tempOrder = draggedElem.reading_order;
        draggedElem.reading_order = targetElem.reading_order;
        targetElem.reading_order = tempOrder;
        renderElements();
    }
}

async function reparseDocument() {
    if (!currentDocId) return;
    if (!confirm('确定要重新解析此文档吗？这将清除现有解析结果。')) return;

    try {
        const res = await fetch(API + '/api/reparse/' + currentDocId, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to start reparsing');
        alert('重解析已开始，请稍候...');
        loadDocumentDetail(currentDocId);
    } catch (e) {
        alert('启动重解析失败: ' + e.message);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

document.addEventListener('DOMContentLoaded', init);

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeEditModal();
    }
});

$('#edit-modal').addEventListener('click', (e) => {
    if (e.target.id === 'edit-modal') {
        closeEditModal();
    }
});
