/**
 * Shared catalog of the MaterialX (.mtlx) sample files that ship alongside
 * this project - used to populate a "Built-in MaterialX" dropdown in a
 * trainer UI (e.g. the website's trainer page) so consumers of this package
 * always offer the same set of sample files instead of hand-maintaining
 * their own, drifting list.
 *
 * Entries are listed alphabetically by file name. `key` is the file name
 * without its .mtlx extension and is stable across releases - safe to use as
 * a <select> option value, an object key, or a URL query parameter.
 *
 * Not every sample is a "training-friendly" tiling material - several of the
 * standard_surface_*_test.mtlx / open_pbr_surface_* / gltf_pbr_* files exist
 * as MaterialX conformance/feature tests rather than curated neural-training
 * targets. They are still fully loadable MeshPhysicalNodeMaterial sources
 * (or, for a couple of shading-model tests, will surface the loader's own
 * "unsupported surface" error, same as opening any other unsupported file
 * from disk) so they're included here for completeness rather than filtered
 * out.
 *
 * NOTE (ported): the original addon shipped these .mtlx files alongside
 * itself (examples/materialx/*.mtlx) and referenced them by bare file name.
 * In this project the actual .mtlx files live in the website package's
 * `public/materialx/` directory instead, so `file` here is deliberately
 * left as a bare file name (no path assumption baked in) - use
 * `getMaterialXSampleUrl(key, root)` below to resolve an actual fetchable
 * URL against whatever root a given deployment serves them from (defaults
 * to `/materialx/`, matching a Vite/Next.js-style `public/materialx/`
 * directory served from the site root).
 */

interface MaterialXSample {
	key: string;
	file: string;
	label: string;
	description: string;
}

export const MATERIALX_SAMPLES: MaterialXSample[] = [
	{
		key: 'gltf_pbr_glass_dispersion',
		file: 'gltf_pbr_glass_dispersion.mtlx',
		label: 'glTF PBR dispersion glass',
		description: 'Rough, near-white glass with chromatic dispersion, built from the gltf_pbr shading model.'
	},
	{
		key: 'alpha_cutoff',
		file: 'alpha_cutoff.mtlx',
		label: 'Alpha cutoff waves',
		description: 'Wavy UV-driven opacity pattern rendered with mask (alpha test) cutoff rather than blending.'
	},
	{
		key: 'brick',
		file: 'brick.mtlx',
		label: 'Brick wall',
		description: 'Procedural brick-and-mortar tiling pattern generated from UV coordinates.'
	},
	{
		key: 'checkerboard',
		file: 'checkerboard.mtlx',
		label: 'Checkerboard 4x4',
		description: 'Simple 4x4 black/white checkerboard base color, a minimal high-frequency tiling test.'
	},
	{
		key: 'checkerboard_normal',
		file: 'checkerboard_normal.mtlx',
		label: 'Checkerboard + normal map',
		description: 'Checkerboard base color combined with a procedural bump/normal pattern.'
	},
	{
		key: 'checkerboard_transparency',
		file: 'checkerboard_transparency.mtlx',
		label: 'Checkerboard transparency',
		description: 'Checkerboard pattern driving blended (non-cutoff) opacity.'
	},
	{
		key: 'emissive_grid',
		file: 'emissive_grid.mtlx',
		label: 'Emissive rainbow waves',
		description: 'Checkerboard-driven emissive rainbow gradient over UV space.'
	},
	{
		key: 'glossy_constant',
		file: 'glossy_constant.mtlx',
		label: 'Glossy constant',
		description: 'Flat, uniform glossy surface with no spatial variation - a minimal specular baseline.'
	},
	{
		key: 'glossy_gold',
		file: 'glossy_gold.mtlx',
		label: 'Metallic gold',
		description: 'Polished metallic gold, a constant-color high-metalness/low-roughness surface.'
	},
	{
		key: 'glossy_red',
		file: 'glossy_red.mtlx',
		label: 'Glossy red',
		description: 'Flat glossy red dielectric surface with visible specular highlight.'
	},
	{
		key: 'lambert_constant',
		file: 'lambert_constant.mtlx',
		label: 'Lambert constant',
		description: 'Flat, uniform matte diffuse surface with no spatial variation - a minimal diffuse baseline.'
	},
	{
		key: 'lambert_red',
		file: 'lambert_red.mtlx',
		label: 'Lambert red',
		description: 'Flat matte red diffuse surface with no specular response.'
	},
	{
		key: 'normal_map',
		file: 'normal_map.mtlx',
		label: 'Normal map waves',
		description: 'Procedural wavy normal-map pattern over a flat base color.'
	},
	{
		key: 'road_aggregate',
		file: 'road_aggregate.mtlx',
		label: 'Road aggregate',
		description: 'Procedural asphalt/aggregate tiling pattern generated from UV coordinates.'
	},
	{
		key: 'uv_grid',
		file: 'uv_grid.mtlx',
		label: 'UV rainbow waves',
		description: 'Smooth rainbow gradient over UV space, a low-frequency default training target.'
	},
	{
		key: 'uv_grid_glossy',
		file: 'uv_grid_glossy.mtlx',
		label: 'UV rainbow glossy',
		description: 'UV rainbow gradient combined with a glossy specular response.'
	},
	{
		key: 'velvet',
		file: 'velvet.mtlx',
		label: 'Velvet',
		description: 'Blue velvet with gold crown embroidery; luminance drives roughness and bump, blueness drives sheen.'
	},
	{
		key: 'open_pbr_surface_honey',
		file: 'open_pbr_surface_honey.mtlx',
		label: 'OpenPBR honey',
		description: 'Amber, translucent honey material built from the open_pbr_surface shading model.'
	},
	{
		key: 'open_pbr_surface_pearl',
		file: 'open_pbr_surface_pearl.mtlx',
		label: 'OpenPBR pearl',
		description: 'Soft, low-roughness pearlescent surface built from the open_pbr_surface shading model.'
	},
	{
		key: 'open_pbr_surface_velvet',
		file: 'open_pbr_surface_velvet.mtlx',
		label: 'OpenPBR velvet',
		description: 'Dark sheen-driven velvet look built from the open_pbr_surface shading model.'
	},
	{
		key: 'standard_surface_color3_vec3_cm_test',
		file: 'standard_surface_color3_vec3_cm_test.mtlx',
		label: 'Color/vector conversion test',
		description: 'MaterialX conformance test exercising color3/vector3 conversions inside a normal-map node graph.'
	},
	{
		key: 'standard_surface_combined_test',
		file: 'standard_surface_combined_test.mtlx',
		label: 'Combined channels test',
		description: 'MaterialX conformance test combining base color, opacity and other standard_surface inputs together.'
	},
	{
		key: 'standard_surface_conditional_if_float',
		file: 'standard_surface_conditional_if_float.mtlx',
		label: 'Conditional (ifgreater) test',
		description: 'MaterialX conformance test driving base color through an ifgreater conditional node graph.'
	},
	{
		key: 'standard_surface_heightnormal',
		file: 'standard_surface_heightnormal.mtlx',
		label: 'Height-to-normal (image) test',
		description: 'MaterialX conformance test converting a height image into a normal map via heighttonormal.'
	},
	{
		key: 'standard_surface_heighttonormal_normal_input',
		file: 'standard_surface_heighttonormal_normal_input.mtlx',
		label: 'Height-to-normal (input) test',
		description: 'MaterialX conformance test feeding a procedural height field into a normal input via heighttonormal.'
	},
	{
		key: 'standard_surface_image_transform',
		file: 'standard_surface_image_transform.mtlx',
		label: 'Image transform (place2d) test',
		description: 'MaterialX conformance test applying a scale/rotate/translate place2d transform to an image texture.'
	},
	{
		key: 'standard_surface_ior_test',
		file: 'standard_surface_ior_test.mtlx',
		label: 'Specular IOR test',
		description: 'MaterialX conformance test of a high, non-metallic specular index of refraction.'
	},
	{
		key: 'standard_surface_opacity_only_test',
		file: 'standard_surface_opacity_only_test.mtlx',
		label: 'Opacity-only test',
		description: 'MaterialX conformance test of blended opacity with no other channel variation.'
	},
	{
		key: 'standard_surface_opacity_test',
		file: 'standard_surface_opacity_test.mtlx',
		label: 'Opacity test',
		description: 'MaterialX conformance test of blended opacity over a colored diffuse surface.'
	},
	{
		key: 'standard_surface_rotate2d_test',
		file: 'standard_surface_rotate2d_test.mtlx',
		label: 'Rotate2D test',
		description: 'MaterialX conformance test rotating a 2D texture coordinate before sampling.'
	},
	{
		key: 'standard_surface_rotate3d_test',
		file: 'standard_surface_rotate3d_test.mtlx',
		label: 'Rotate3D test',
		description: 'MaterialX conformance test of a 3D rotation node graph feeding base color.'
	},
	{
		key: 'standard_surface_rotate_scale2d_test',
		file: 'standard_surface_rotate_scale2d_test.mtlx',
		label: 'Rotate+Scale2D test',
		description: 'MaterialX conformance test chaining rotate2d then a non-uniform scale before sampling - the opposite node order from scale_rotate2d_test.mtlx.'
	},
	{
		key: 'standard_surface_roughness_test',
		file: 'standard_surface_roughness_test.mtlx',
		label: 'Roughness map test',
		description: 'MaterialX conformance test driving specular roughness from a procedural map.'
	},
	{
		key: 'standard_surface_scale2d_test',
		file: 'standard_surface_scale2d_test.mtlx',
		label: 'Scale2D test',
		description: 'MaterialX conformance test applying a non-uniform 2D scale to a texture coordinate before sampling.'
	},
	{
		key: 'standard_surface_scale_rotate2d_test',
		file: 'standard_surface_scale_rotate2d_test.mtlx',
		label: 'Scale+Rotate2D test',
		description: 'MaterialX conformance test chaining a non-uniform scale then rotate2d before sampling - the opposite node order from rotate_scale2d_test.mtlx.'
	},
	{
		key: 'standard_surface_sheen_test',
		file: 'standard_surface_sheen_test.mtlx',
		label: 'Sheen test',
		description: 'MaterialX conformance test of the sheen/sheen_color/sheen_roughness inputs over a blue base.'
	},
	{
		key: 'standard_surface_specular_test',
		file: 'standard_surface_specular_test.mtlx',
		label: 'Specular test',
		description: 'MaterialX conformance test of the specular/specular_color inputs over a blue base.'
	},
	{
		key: 'standard_surface_texture_opacity_test',
		file: 'standard_surface_texture_opacity_test.mtlx',
		label: 'Texture opacity test',
		description: 'MaterialX conformance test driving opacity from a procedural texture graph.'
	},
	{
		key: 'standard_surface_thin_film_ior_clamp_test',
		file: 'standard_surface_thin_film_ior_clamp_test.mtlx',
		label: 'Thin film IOR clamp test',
		description: 'MaterialX conformance test of thin-film interference with a clamped, high thin-film IOR.'
	},
	{
		key: 'standard_surface_thin_film_rainbow_test',
		file: 'standard_surface_thin_film_rainbow_test.mtlx',
		label: 'Thin film rainbow test',
		description: 'MaterialX conformance test of thin-film interference producing an iridescent rainbow highlight.'
	},
	{
		key: 'standard_surface_transmission_only_test',
		file: 'standard_surface_transmission_only_test.mtlx',
		label: 'Transmission-only test',
		description: 'MaterialX conformance test of full transmission (clear glass) with no roughness.'
	},
	{
		key: 'standard_surface_transmission_rough',
		file: 'standard_surface_transmission_rough.mtlx',
		label: 'Rough transmission test',
		description: 'MaterialX conformance test of frosted, near-full transmission with visible surface roughness.'
	},
	{
		key: 'standard_surface_transmission_test',
		file: 'standard_surface_transmission_test.mtlx',
		label: 'Transmission test',
		description: 'MaterialX conformance test of high, slightly tinted transmission (clear glass/liquid).'
	},
	{
		key: 'wood',
		file: 'wood.mtlx',
		label: 'Hardwood floor',
		description: 'Tiled hardwood floor built from the webgpu_lights_physical.html diffuse/bump/roughness bitmaps.'
	}
];

export function getMaterialXSample( key: string ): MaterialXSample | undefined {

	return MATERIALX_SAMPLES.find( ( sample ) => sample.key === key );

}

// Default root a sample's bare `file` name is resolved against - matches a
// `public/materialx/` directory served from the site root (Vite/Next.js
// convention). Override via the `root` argument for a different deployment
// layout.
const DEFAULT_MATERIALX_ROOT = '/materialx/';

/**
 * Resolves a catalog entry's bare `file` name into a fetchable URL. See this
 * module's doc comment for why `file` itself carries no path assumption.
 */
export function getMaterialXSampleUrl( key: string, root: string = DEFAULT_MATERIALX_ROOT ): string | undefined {

	const sample = getMaterialXSample( key );
	if ( ! sample ) return undefined;

	return `${ root }${ sample.file }`;

}

/**
 * Populates a <select> element with one <option> per sample (in the
 * catalog's alphabetical-by-file order) followed by a trailing, disabled
 * "Custom file" option - matching the pattern the original addon's neural-*
 * examples used for their "Open .mtlx" file picker.
 */
export function populateMaterialXSelect( selectElement: HTMLSelectElement, options: { customLabel?: string } = {} ): void {

	const customLabel = options.customLabel || 'Custom file';

	for ( const sample of MATERIALX_SAMPLES ) {

		const option = document.createElement( 'option' );
		option.value = sample.key;
		option.textContent = sample.label;
		selectElement.appendChild( option );

	}

	const customOption = document.createElement( 'option' );
	customOption.value = '';
	customOption.disabled = true;
	customOption.textContent = customLabel;
	selectElement.appendChild( customOption );

}
