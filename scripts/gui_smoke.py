"""Launch the real local server, exercise the browser and capture actual results.

Requires Playwright only for development / CI, never for app installation.
"""
from pathlib import Path
import json
import sys
import threading

from playwright.sync_api import sync_playwright, expect

from fiatlux_app.model import examples
from fiatlux_app.server import make_server


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else 'gui-captures')
    output.mkdir(parents=True, exist_ok=True)
    server = make_server(0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport=dict(width=1440, height=1080), device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/linear')
            expect(page.locator('.block')).to_have_count(4)
            for key, config in examples().items():
                page.locator('#examples').select_option(key)
                expect(page.locator('#project-name')).to_have_value(config['name'])
                page.locator('#run').click()
                expect(page.locator('#result-state')).to_have_text('À jour', timeout=120000)
                expect(page.locator('#image')).to_be_visible()
                expect(page.locator('#image')).to_have_attribute('data-run', server.manager.run_id)
                expect(page.locator('#message.error')).to_have_count(0)
                assert page.locator('#stage option').count() == len(config['blocks'])
                assert page.locator('#range').inner_text() != '—'
                # Confirm browser transport carries actual nonzero FIATLUX data.
                result = server.manager.future.result(timeout=2)
                image = result.image(len(config['blocks'])-1)
                assert image['maximum'] > 0
                assert image['unit'] == 'e⁻/pixel'
                page.screenshot(path=str(output / f'{key}.png'), full_page=True)
                if key == '02_elt_dm':
                    page.get_by_role('button', name='Sélectionner Miroir déformable · Zernike · dm').click()
                    page.locator('#stage').select_option('2')
                    page.locator('#quantity').select_option('opd')
                    expect(page.locator('#image-unit')).to_have_text('nm')
                    assert result.image(2, 'opd')['maximum'] > 0
                    page.screenshot(path=str(output / '02_dm_opd.png'), full_page=True)
                page.locator('#stage').select_option('1')
                page.locator('#quantity').select_option('phase')
                expect(page.locator('#channel')).to_have_value('0')
                expect(page.locator('#image-unit')).to_have_text('rad')
                # Save through the UI, reload through the validated file input.
                with page.expect_download() as download:
                    page.locator('#save').click()
                saved = output / f'{key}.json'
                download.value.save_as(saved)
                assert json.loads(saved.read_text()) == config
                page.locator('#file').set_input_files(saved)
                expect(page.locator('#result-state')).to_have_text('Résultats périmés')
            # Three real spectral channels; detector remains integrated.
            page.locator('#examples').select_option('01_circular')
            page.locator('#system-tab').click()
            page.locator('#spectrum-samples').fill('3')
            page.locator('#spectrum-samples').dispatch_event('change')
            page.locator('#spectrum-delta_wavelength').fill('1e-7')
            page.locator('#spectrum-delta_wavelength').dispatch_event('change')
            page.locator('#run').click()
            expect(page.locator('#result-state')).to_have_text('À jour', timeout=120000)
            expect(page.locator('#image')).to_have_attribute('data-run', server.manager.run_id)
            assert page.locator('#channel option:disabled').count() == 3
            page.locator('#stage').select_option('2')
            page.locator('#channel').select_option('1')
            expect(page.locator('#image-unit')).to_have_text('photons/s/m²')
            page.locator('#log').click()
            page.screenshot(path=str(output/'polychromatic.png'), full_page=True)
            page.locator('#quantity').select_option('phase')
            expect(page.locator('#image-unit')).to_have_text('rad')
            page.locator('#quantity').select_option('intensity')
            page.locator('#linear').click()
            # Invalid unparsed inputs must never execute the previous valid value.
            page.locator('#examples').select_option('02_elt_dm')
            page.get_by_role('button', name='Sélectionner Miroir déformable · Zernike · dm').click()
            page.locator('#dm-commands').fill('[invalid]')
            page.locator('#dm-commands').dispatch_event('change')
            previous_run = server.manager.run_id
            page.locator('#run').click()
            expect(page.locator('#message')).to_contain_text('dm')
            assert server.manager.run_id == previous_run
            page.locator('#system-tab').click()
            page.locator('#block-tab').click()
            expect(page.locator('#dm-commands')).to_have_value('[invalid]')
            # Parameter edits, reorder and invalid configuration retain user data.
            page.locator('#examples').select_option('01_circular')
            page.get_by_role('button', name='Sélectionner Pupille circulaire · pupil').click()
            page.locator('#pupil-radius').fill('0.4')
            page.locator('#pupil-radius').dispatch_event('change')
            page.locator('#run').click()
            expect(page.locator('#result-state')).to_have_text('À jour', timeout=120000)
            page.locator('#pupil-radius').fill('-1')
            page.locator('#pupil-radius').dispatch_event('change')
            expect(page.locator('#result-state')).to_have_text('Résultats périmés')
            page.locator('#run').click()
            expect(page.locator('#message')).to_contain_text('pupil', timeout=10000)
            expect(page.locator('#pupil-radius')).to_have_value('-1')
            expect(page.locator('#run')).to_be_enabled()
            page.screenshot(path=str(output/'validation-error.png'), full_page=True)
            page.locator('#pupil-radius').fill('0.5')
            page.locator('#pupil-radius').dispatch_event('change')
            page.get_by_role('button', name='Après →').click()
            page.locator('#validate').click()
            expect(page.locator('#message')).to_contain_text('grille incompatible')
            page.get_by_role('button', name='← Avant').click()
            page.locator('#search').fill('Kolmogorov')
            page.locator('.library-item').click()
            expect(page.locator('.block')).to_have_count(5)
            page.get_by_role('button', name='Supprimer', exact=True).click()
            expect(page.locator('.block')).to_have_count(4)
            page.set_viewport_size(dict(width=390, height=844))
            page.screenshot(path=str(output/'mobile.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server.manager.close()
        server.canvas_manager.close()
        worker.join(timeout=5)
    print('Real FIATLUX browser smoke passed: examples, spectral views, OPD, save/reload, stale state, edit/order/add/delete, errors and mobile.')


if __name__ == '__main__':
    main()
