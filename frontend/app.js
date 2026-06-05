const API = '';

let currentDocId = null;
let pollTimer = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function init() {
    setupDropZone();
    setupTabs();
    loadDocuments();
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
        currentDocId = result.document_id;
        loadDocuments();
        startParsing(result.document_id);
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
        renderDocuments(data.documents);
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

function renderDocuments(docs) {
    const container = $('#documents-list');
    if (!docs || docs.length === 0) {
        container.innerHTML = '<p class="empty-msg">暂无文档</p>';
        return;
    }

    container.innerHTML = docs.map(doc => `
        <div class="doc-item">
            <div class="doc-info">
                <div class="doc-name">${escapeHtml(doc.original_filename)}</div>
                <div class="doc-meta">
                    ID: ${doc.id} | ${doc.page_count} 页 |
                    <span class="status-badge status-${doc.status}">${doc.status}</span>
                </div>
            </div>
            <div class="doc-actions">
                ${doc.status !== 'processing' ? `<button class="btn btn-primary" onclick="startParsing(${doc.id})">解析</button>` : ''}
                ${doc.status === 'completed' ? `<button class="btn btn-success" onclick="viewResults(${doc.id})">查看结果</button>` : ''}
                <button class="btn btn-outline" onclick="viewStatus(${doc.id})">状态</button>
                <button class="btn btn-danger" onclick="deleteDocument(${doc.id})">删除</button>
            </div>
        </div>
    `).join('');
}

async function startParsing(docId) {
    currentDocId = docId;
    try {
        const res = await fetch(API + '/api/parse/' + docId, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to start parsing');

        $('#parse-section').classList.remove('hidden');
        startPolling(docId);
    } catch (e) {
        alert('启动解析失败: ' + e.message);
    }
}

function startPolling(docId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollStatus(docId), 2000);
    pollStatus(docId);
}

async function pollStatus(docId) {
    try {
        const res = await fetch(API + '/api/status/' + docId);
        const data = await res.json();

        let statusText = `文档状态: <span class="status-badge status-${data.status}">${data.status}</span>`;
        $('#parse-status').innerHTML = statusText;

        const pagesContainer = $('#parse-pages');
        pagesContainer.innerHTML = data.pages.map(p => `
            <div class="page-card">
                <div class="page-num">第 ${p.page_number} 页</div>
                <div class="page-status">
                    <span class="status-badge status-${p.status}">${p.status}</span>
                    ${p.is_scanned ? ' (扫描页)' : ''}
                </div>
            </div>
        `).join('');

        if (data.status === 'completed') {
            clearInterval(pollTimer);
            pollTimer = null;
            viewResults(docId);
        } else if (data.status === 'failed') {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    } catch (e) {
        console.error('Poll status failed:', e);
    }
}

async function viewStatus(docId) {
    currentDocId = docId;
    $('#parse-section').classList.remove('hidden');
    startPolling(docId);
}

async function viewResults(docId) {
    currentDocId = docId;
    try {
        const res = await fetch(API + '/api/results/' + docId);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to load results');

        $('#results-section').classList.remove('hidden');

        $('#markdown-content').textContent = data.markdown || '(无内容)';

        renderElements(data.pages);

        $('#json-content').textContent = JSON.stringify(data, null, 2);

        const resultsEl = document.getElementById('results-section');
        resultsEl.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        alert('加载结果失败: ' + e.message);
    }
}

function renderElements(pages) {
    const container = $('#elements-content');
    let html = '';

    for (const page of pages) {
        html += `<h3 style="margin: 1rem 0 0.5rem;">第 ${page.page_number} 页 ${page.is_scanned ? '(扫描页)' : ''}</h3>`;

        if (!page.elements || page.elements.length === 0) {
            html += '<p style="color: #64748b;">无解析元素</p>';
            continue;
        }

        const sorted = [...page.elements].sort((a, b) => a.reading_order - b.reading_order);

        for (const elem of sorted) {
            const isLatex = elem.content_format === 'latex';
            const isImage = elem.content_format === 'image_path';

            let contentHtml = '';
            if (isImage && elem.content) {
                const relPath = elem.content.replace(/\\/g, '/').split('/').slice(-3).join('/');
                contentHtml = `<img src="/api/file/${encodeURIComponent(elem.content)}" alt="Picture">`;
            } else {
                contentHtml = escapeHtml(elem.content || '(空)');
            }

            html += `
                <div class="element-item">
                    <div class="element-header">
                        <span class="element-type">${escapeHtml(elem.type)}</span>
                        <span class="element-order">顺序: ${elem.reading_order}</span>
                        <span class="element-bbox">bbox: [${elem.bbox.map(v => v.toFixed(1)).join(', ')}]</span>
                        <span class="element-bbox">置信度: ${(elem.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="element-content ${isLatex ? 'latex' : ''}">${contentHtml}</div>
                </div>
            `;
        }
    }

    container.innerHTML = html;
}

async function deleteDocument(docId) {
    if (!confirm('确定要删除此文档及其解析结果吗？')) return;

    try {
        await fetch(API + '/api/documents/' + docId, { method: 'DELETE' });
        loadDocuments();
        if (currentDocId === docId) {
            $('#parse-section').classList.add('hidden');
            $('#results-section').classList.add('hidden');
            currentDocId = null;
        }
    } catch (e) {
        alert('删除失败: ' + e.message);
    }
}

function setupTabs() {
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.tab-btn').forEach(b => b.classList.remove('active'));
            $$('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const tab = btn.dataset.tab;
            $(`#tab-${tab}`).classList.add('active');
        });
    });
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

document.addEventListener('DOMContentLoaded', init);
