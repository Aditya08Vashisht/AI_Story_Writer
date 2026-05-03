const http = require('http');
const fs = require('fs');
const path = require('path');

const root = __dirname;
const port = Number(process.env.PORT || 5051);

const types = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
};

function send(res, status, body, type = 'text/plain; charset=utf-8') {
    res.writeHead(status, {
        'Content-Type': type,
        'Cache-Control': 'no-store',
    });
    res.end(body);
}

function resolveFile(urlPath) {
    const requested = urlPath === '/' ? '/demo.html' : urlPath;
    const decoded = decodeURIComponent(requested.split('?')[0]);
    const filePath = path.normalize(path.join(root, decoded));
    if (!filePath.startsWith(root)) return null;
    return filePath;
}

const server = http.createServer((req, res) => {
    const filePath = resolveFile(req.url || '/');
    if (!filePath) {
        send(res, 403, 'Forbidden');
        return;
    }

    fs.readFile(filePath, (error, data) => {
        if (error) {
            send(res, 404, 'Not found');
            return;
        }

        const type = types[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
        send(res, 200, data, type);
    });
});

server.listen(port, '127.0.0.1', () => {
    console.log(`Story Studio static demo running at http://127.0.0.1:${port}`);
});
