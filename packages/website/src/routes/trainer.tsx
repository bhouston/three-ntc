import { createFileRoute } from '@tanstack/react-router';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const Route = createFileRoute('/trainer')({
  component: TrainerPage,
});

// Friendly labels derived from the filenames in public/materialx/.
const EXAMPLE_MATERIALS = [
  { value: '/materialx/alpha_cutoff.mtlx', label: 'Alpha Cutoff' },
  { value: '/materialx/brick.mtlx', label: 'Brick' },
  { value: '/materialx/checkerboard_normal.mtlx', label: 'Checkerboard Normal' },
  { value: '/materialx/checkerboard_transparency.mtlx', label: 'Checkerboard Transparency' },
  { value: '/materialx/checkerboard.mtlx', label: 'Checkerboard' },
  { value: '/materialx/emissive_grid.mtlx', label: 'Emissive Grid' },
  { value: '/materialx/glossy_constant.mtlx', label: 'Glossy Constant' },
  { value: '/materialx/glossy_gold.mtlx', label: 'Glossy Gold' },
  { value: '/materialx/glossy_red.mtlx', label: 'Glossy Red' },
  { value: '/materialx/gltf_pbr_glass_dispersion.mtlx', label: 'glTF PBR Glass Dispersion' },
  { value: '/materialx/lambert_constant.mtlx', label: 'Lambert Constant' },
  { value: '/materialx/lambert_red.mtlx', label: 'Lambert Red' },
  { value: '/materialx/normal_map.mtlx', label: 'Normal Map' },
  { value: '/materialx/open_pbr_surface_honey.mtlx', label: 'OpenPBR Surface Honey' },
  { value: '/materialx/open_pbr_surface_pearl.mtlx', label: 'OpenPBR Surface Pearl' },
  { value: '/materialx/open_pbr_surface_velvet.mtlx', label: 'OpenPBR Surface Velvet' },
  { value: '/materialx/road_aggregate.mtlx', label: 'Road Aggregate' },
  { value: '/materialx/standard_surface_color3_vec3_cm_test.mtlx', label: 'Standard Surface Color3 Vec3 CM Test' },
  { value: '/materialx/standard_surface_combined_test.mtlx', label: 'Standard Surface Combined Test' },
  { value: '/materialx/standard_surface_conditional_if_float.mtlx', label: 'Standard Surface Conditional If Float' },
  { value: '/materialx/standard_surface_heightnormal.mtlx', label: 'Standard Surface Height Normal' },
  {
    value: '/materialx/standard_surface_heighttonormal_normal_input.mtlx',
    label: 'Standard Surface Height To Normal Input',
  },
  { value: '/materialx/standard_surface_image_transform.mtlx', label: 'Standard Surface Image Transform' },
  { value: '/materialx/standard_surface_ior_test.mtlx', label: 'Standard Surface IOR Test' },
  { value: '/materialx/standard_surface_opacity_only_test.mtlx', label: 'Standard Surface Opacity Only Test' },
  { value: '/materialx/standard_surface_opacity_test.mtlx', label: 'Standard Surface Opacity Test' },
  { value: '/materialx/standard_surface_rotate_scale2d_test.mtlx', label: 'Standard Surface Rotate Scale2D Test' },
  { value: '/materialx/standard_surface_rotate2d_test.mtlx', label: 'Standard Surface Rotate2D Test' },
  { value: '/materialx/standard_surface_rotate3d_test.mtlx', label: 'Standard Surface Rotate3D Test' },
  { value: '/materialx/standard_surface_roughness_test.mtlx', label: 'Standard Surface Roughness Test' },
  { value: '/materialx/standard_surface_scale_rotate2d_test.mtlx', label: 'Standard Surface Scale Rotate2D Test' },
  { value: '/materialx/standard_surface_scale2d_test.mtlx', label: 'Standard Surface Scale2D Test' },
  { value: '/materialx/standard_surface_sheen_test.mtlx', label: 'Standard Surface Sheen Test' },
  { value: '/materialx/standard_surface_specular_test.mtlx', label: 'Standard Surface Specular Test' },
  { value: '/materialx/standard_surface_texture_opacity_test.mtlx', label: 'Standard Surface Texture Opacity Test' },
  {
    value: '/materialx/standard_surface_thin_film_ior_clamp_test.mtlx',
    label: 'Standard Surface Thin Film IOR Clamp Test',
  },
  { value: '/materialx/standard_surface_thin_film_rainbow_test.mtlx', label: 'Standard Surface Thin Film Rainbow Test' },
  { value: '/materialx/standard_surface_transmission_only_test.mtlx', label: 'Standard Surface Transmission Only Test' },
  { value: '/materialx/standard_surface_transmission_rough.mtlx', label: 'Standard Surface Transmission Rough' },
  { value: '/materialx/standard_surface_transmission_test.mtlx', label: 'Standard Surface Transmission Test' },
  { value: '/materialx/uv_grid_glossy.mtlx', label: 'UV Grid Glossy' },
  { value: '/materialx/uv_grid.mtlx', label: 'UV Grid' },
  { value: '/materialx/velvet.mtlx', label: 'Velvet' },
  { value: '/materialx/wood.mtlx', label: 'Wood' },
];

function TrainerPage() {
  return (
    <div className="flex flex-1 flex-col gap-4 p-4">
      <div>
        <h1 className="text-lg font-semibold">MaterialX Trainer</h1>
        <p className="text-sm text-muted-foreground">
          Pick a MaterialX example to bake into a .ntc model. Training UI coming soon.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {EXAMPLE_MATERIALS.map((m) => (
          <Card key={m.value}>
            <CardHeader>
              <CardTitle className="text-sm">{m.label}</CardTitle>
            </CardHeader>
            <CardContent>
              <Button size="sm" variant="outline" disabled className="w-full">
                Train (coming soon)
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
