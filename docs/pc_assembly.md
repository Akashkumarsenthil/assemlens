# Future application: PC assembly and HP hardware upgrades

A customer buys compatible computer components, scans a QR code, loads the exact product/manual package, and points a camera at the workspace. AssemLens could observe supported steps, flag visible mismatches, and ask for confirmation or another angle. It could cover a documented HP upgrade procedure or a custom PC build using documented compatible parts.

This is a planned extension, not a trained or validated capability today. HP desktops can have model-specific parts, connectors and service procedures. Do not assume arbitrary retail components work in every HP PC. Use the exact model's official service documentation and an explicit compatibility list. No HP endorsement is implied.

Candidate visible checks: RAM orientation and visible latches, GPU bracket/retention placement, and cable placement when connectors and their documentation are clearly visible. Use powered-off equipment and product-specific service instructions. Do not open a power supply. Camera output cannot certify electrical safety, hidden contact, torque, thermal performance or a successful boot.

The example JSON is a design template with no reference images; it must not be enabled as a supported product. Add reviewed reference images, exact manual URL, supported steps, known errors, and a held-out evaluation before enabling it. The toy-action adapter is not evidence of PC assembly accuracy.

For the first hackathon demo, one safely repeatable, visible task is enough. Record error -> correction -> verification, plus an uncertain view and the cloud-disabled fallback. Label precomputed predictions and report measured inference time.
