import asyncio
from aiohttp import web
from app import app

async def run_web_app():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0')
    print(f"Web app running on http://localhost:8080")
    await site.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_web_app())

