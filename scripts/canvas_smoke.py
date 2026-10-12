"""Functional and visual checks on a real localhost FIATLUX canvas."""
import json
from pathlib import Path
import sys
import threading
import time

from playwright.sync_api import sync_playwright, expect
from fiatlux_app.server import make_server
from fiatlux_app.graph import compile_graph


def main():
    output=Path(sys.argv[1] if len(sys.argv)>1 else 'canvas-captures')
    output.mkdir(parents=True,exist_ok=True)
    server=make_server(0)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport=dict(width=1680,height=1080))
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            expect(page.locator('#state')).to_have_text('À jour',timeout=120000)
            expect(page.locator('.node')).to_have_count(9)
            for bid in ('view-phase','view-opd','view-intensity'):
                expect(page.locator(f'[data-view="{bid}"]')).to_have_attribute('data-revision','1',timeout=20000)
            session=page.evaluate('session')
            engine=server.canvas_manager.engines[session]
            first=server.canvas_manager.sessions[session].result
            assert first.snapshots['focus'].field.intensity().max()>0
            page.screenshot(path=str(output/'01-canvas.png'),full_page=True)
            # Free placement must not change topology, revision or execute anything.
            old_graph=page.evaluate('config')
            old_count=engine.evaluations
            head=page.locator('[data-node="dm"] .node-head').bounding_box()
            page.mouse.move(head['x']+30,head['y']+25)
            page.mouse.down();page.mouse.move(head['x']+90,head['y']-20,steps=10);page.mouse.up()
            page.wait_for_timeout(600)
            assert engine.evaluations==old_count
            moved=page.evaluate('config')
            assert moved['layout']['positions']['dm']!=old_graph['layout']['positions']['dm']
            assert moved['connections']==old_graph['connections']
            # A display change is local: no native propagation or new atmosphere.
            page.locator('[data-node="view-phase"] .node-head').click()
            page.locator('#view-phase-colormap').select_option('magma')
            page.wait_for_timeout(500)
            assert engine.evaluations==old_count
            assert server.canvas_manager.sessions[session].result is first
            page.get_by_role('button',name='Agrandir view-phase').click()
            expect(page.locator('#expanded')).to_be_visible()
            page.screenshot(path=str(output/'02-expanded.png'),full_page=True)
            page.locator('#close-expanded').click()
            # DM invalidation must preserve native upstream snapshots and operators.
            page.locator('[data-node="dm"] .node-head').click()
            upstream={key:engine.entries[key] for key in ('source','pupil','atmosphere')}
            page.locator('#dm-commands').fill('[4e-8,-1.5e-8,6e-8]')
            expect(page.locator('#state')).to_have_text('À jour',timeout=120000)
            page.wait_for_function('published === revision && revision > 1')
            changed=server.canvas_manager.sessions[session].result
            assert changed.computed==['dm','focus','detector']
            assert all(engine.entries[key] is value for key,value in upstream.items())
            screen=changed.snapshots['atmosphere'].opd.clone()
            # Invalid scientific input preserves last map, does not execute.
            old_count=engine.evaluations
            page.locator('#dm-commands').fill('[invalid]')
            expect(page.locator('#state')).to_have_text('Erreur',timeout=10000)
            assert engine.evaluations==old_count
            assert page.locator('[data-view="view-opd"]').get_attribute('data-revision')
            page.screenshot(path=str(output/'03-invalid.png'),full_page=True)
            page.locator('#dm-commands').fill('[4e-8,-1.5e-8,6e-8]')
            page.wait_for_function('published === revision && state === "current"',timeout=120000)
            # Explicit stochastic refresh; display had not resampled anything.
            page.locator('[data-node="atmosphere"] .node-head').click()
            page.locator('#new-realization').click()
            page.wait_for_function('published === revision && state === "current"',timeout=120000)
            changed=server.canvas_manager.sessions[session].result
            assert not __import__('torch').equal(changed.snapshots['atmosphere'].opd,screen)
            # Real palette drag/drop, then actual typed port connections.
            before=len(page.evaluate('config.nodes'))
            card=page.locator('.palette-item[data-kind="Visualization"]')
            box=page.locator('#viewport').bounding_box()
            card.dispatch_event('dragstart',{'dataTransfer':page.evaluate_handle('new DataTransfer()')})
            transfer=page.evaluate_handle('new DataTransfer()')
            transfer.evaluate('(t)=>t.setData("application/x-fiatlux-component","Visualization")')
            page.locator('#viewport').dispatch_event('drop',{'dataTransfer':transfer,'clientX':box['x']+300,'clientY':box['y']+220})
            expect(page.locator('.node')).to_have_count(before+1)
            new_id=page.evaluate('config.nodes.at(-1).id')
            old_count=engine.evaluations
            # Click output and input ports (also available as native pointer drag).
            page.get_by_role('button',name='pupil output field Field',exact=True).click()
            page.get_by_role('button',name=f'{new_id} input field Field',exact=True).click()
            page.wait_for_function('(id)=>config.connections.some(e=>e.to.node===id)',arg=new_id)
            assert engine.evaluations==old_count
            page.wait_for_function('(id)=>maps.has(id)',arg=new_id)
            assert server.canvas_manager.sessions[session].result is changed
            # Refuse an incompatible port and preserve all existing connections.
            connections=page.evaluate('config.connections.length')
            page.get_by_role('button',name='dm output opd OPD',exact=True).click()
            page.get_by_role('button',name='focus input in Field',exact=True).click()
            expect(page.locator('#message')).to_contain_text('ports incompatibles')
            assert page.evaluate('config.connections.length')==connections
            # Save/reload positions and connections; this must remain a cache hit.
            with page.expect_download() as download:
                page.locator('#save').click()
            saved=output/'canvas-v2.json';download.value.save_as(saved)
            saved_graph=json.loads(saved.read_text())
            assert saved_graph==page.evaluate('config')
            assert compile_graph(saved_graph)['blocks'][-1]['id']=='detector'
            page.locator('#file').set_input_files(saved)
            page.wait_for_timeout(600)
            assert engine.evaluations==old_count
            assert page.evaluate('config.layout')==saved_graph['layout']
            page.locator('[data-node="dm"] .node-head').hover()
            expect(page.locator('#tooltip')).to_be_visible()
            page.screenshot(path=str(output/'04-tooltip.png'),full_page=True)
            page.mouse.move(10,10)
            # Zoom and pan do not run science; selection/delete are functional.
            page.locator('#zoom-in').click();page.locator('#zoom-out').click()
            page.locator('#fit').click()
            assert engine.evaluations==old_count
            page.locator(f'[data-node="{new_id}"] .node-head').click()
            page.locator('#delete').click()
            expect(page.locator('.node')).to_have_count(9)
            assert engine.evaluations==old_count
            page.screenshot(path=str(output/'05-final.png'),full_page=True)
            page.set_viewport_size(dict(width=390,height=844));page.locator('#fit').click()
            page.screenshot(path=str(output/'06-mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert not errors,errors
            browser.close()
    finally:
        server.shutdown();server.server_close();server.manager.close();server.canvas_manager.close();thread.join(timeout=5)
    print('Canvas browser smoke passed: drag/drop, move, typed connections, edit, incremental DM, stale/error recovery, views, atmosphere refresh, save/load, zoom, tooltip, delete, mobile.')


if __name__=='__main__':main()
