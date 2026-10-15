# Public face inference fixture

`astronaut.png` is the 512 x 512 NASA photograph of astronaut Eileen Collins,
retrieved unchanged from scikit-image's official `v0.25.2` release:

- [Pinned release asset](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png)
- [Official scikit-image data provenance and public-domain notice](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)
- Original credit: NASA; the scikit-image documentation identifies the photograph as public domain with no known copyright restrictions.
- SHA-256: `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`
- Size: 791,555 bytes.

Tests use this single public photograph to check actual model loading, alignment,
finite normalized embeddings, deterministic repeat extraction, and rejection of
two faces composed into one image. It is not an independent biometric dataset,
a liveness sample, an accuracy study, or an endorsement by NASA/the subject.
No user webcam photographs are included in this directory.

## Second independent portrait

`grace_hopper.jpg` is the official U.S. Navy portrait of Commodore Grace M.
Hopper (photographer James S. Davis, 20 January 1984), as distributed in
Matplotlib's official `v3.10.0` sample data. It was retrieved unchanged:

- [Pinned Matplotlib release asset](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg)
- [Matplotlib's official example using this sample](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html)
- Original photograph credit: James S. Davis / U.S. Navy, photo ID `DN-SC-84-05971` (covered portrait).
- [U.S. Navy archive copyright policy for official government photographs](https://www.history.navy.mil/about-us/services-and-policies/scan-copyright.html)
- [U.S. Department of Defense image public-domain guidance](https://www.defense.gov/Contact/Help-Center/Help-Center-Listing/Tag/130833/photos/)
- SHA-256: `a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130`
- Size: 61,306 bytes; 512 x 600 pixels.

The pair smoke test compares the NASA portrait with a mild in-memory brightness
and two-degree rotation of that same photograph, and separately with the Navy
portrait. It preserves the pre-existing OpenCV example threshold of 0.363;
the threshold is not fitted to these fixtures. The first comparison is a
same-source transformation, **not an independent same-person camera capture**.
Two photographs do not support claims about biometric accuracy, false-match
rates, demographic fairness, spoof resistance or subject/government endorsement.
