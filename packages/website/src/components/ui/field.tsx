import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

// A pared-down port of shadcn's Field primitives - just the pieces the
// trainer form actually uses (Field/FieldGroup/FieldLabel/FieldDescription/
// FieldError, plus FieldSet/FieldLegend for grouping sections).

function FieldSet({ className, ...props }: React.ComponentProps<'fieldset'>) {
  return (
    <fieldset
      data-slot="field-set"
      className={cn('flex flex-col gap-4 rounded-lg border border-border p-4', className)}
      {...props}
    />
  );
}

function FieldLegend({ className, ...props }: React.ComponentProps<'legend'>) {
  return (
    <legend
      data-slot="field-legend"
      className={cn('-ml-1 px-1 text-sm font-medium text-foreground', className)}
      {...props}
    />
  );
}

function FieldGroup({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="field-group" className={cn('flex flex-col gap-4', className)} {...props} />;
}

const fieldVariants = cva('flex gap-2', {
  variants: {
    orientation: {
      vertical: 'flex-col',
      horizontal: 'flex-row items-center justify-between',
    },
  },
  defaultVariants: {
    orientation: 'vertical',
  },
});

function Field({
  className,
  orientation,
  'data-invalid': dataInvalid,
  ...props
}: React.ComponentProps<'div'> & VariantProps<typeof fieldVariants> & { 'data-invalid'?: boolean }) {
  return (
    <div
      data-slot="field"
      data-invalid={dataInvalid || undefined}
      className={cn(fieldVariants({ orientation }), className)}
      {...props}
    />
  );
}

function FieldLabel({ className, ...props }: React.ComponentProps<'label'>) {
  return (
    <label
      data-slot="field-label"
      className={cn('text-sm leading-none font-medium group-data-[disabled=true]:opacity-50', className)}
      {...props}
    />
  );
}

function FieldContent({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="field-content" className={cn('flex flex-col gap-1', className)} {...props} />;
}

function FieldTitle({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="field-title" className={cn('text-sm font-medium', className)} {...props} />;
}

function FieldDescription({ className, ...props }: React.ComponentProps<'p'>) {
  return <p data-slot="field-description" className={cn('text-xs text-muted-foreground', className)} {...props} />;
}

function FieldError({
  errors,
  className,
  ...props
}: React.ComponentProps<'p'> & { errors?: Array<{ message?: string } | undefined> | undefined }) {
  const message = errors?.find((error) => error?.message)?.message;
  if (!message) return null;

  return (
    <p data-slot="field-error" className={cn('text-xs text-destructive', className)} {...props}>
      {message}
    </p>
  );
}

export { Field, FieldContent, FieldDescription, FieldError, FieldGroup, FieldLabel, FieldLegend, FieldSet, FieldTitle };
