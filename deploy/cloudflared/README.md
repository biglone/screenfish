# Cloudflare Tunnel (示例)

推荐用 Cloudflare Tunnel 将服务暴露到公网：应用只监听 `127.0.0.1:8000`，服务器不需要开放 8000 端口。

## 1) 在云服务器安装并登录 cloudflared

按 Cloudflare 官方文档安装 `cloudflared`，然后：

```bash
cloudflared tunnel login
```

## 2) 创建 Tunnel 并绑定域名

```bash
cloudflared tunnel create stock-screener
cloudflared tunnel route dns stock-screener api.your-domain.com
```

## 3) 配置并运行

创建配置 `~/.cloudflared/config.yml`，示例：

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /root/.cloudflared/<TUNNEL-UUID>.json

ingress:
  - hostname: api.your-domain.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

运行：

```bash
cloudflared tunnel run stock-screener
```

## 安全建议

- 设置 `STOCK_SCREENER_API_KEY` 并在客户端带 `X-API-Key` 请求头
- 用 Cloudflare WAF/Rate Limit 保护 API
- Web 端跨域访问时设置 `STOCK_SCREENER_CORS_ORIGINS`

