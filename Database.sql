-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.analysis_history (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL,
  product_id uuid,
  credits_used integer NOT NULL DEFAULT 1,
  analysis_type text NOT NULL DEFAULT 'full_optimization'::text,
  status text DEFAULT 'completed'::text CHECK (status = ANY (ARRAY['pending'::text, 'completed'::text, 'failed'::text])),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  google_analysis_id uuid,
  perplexity_analysis_id uuid,
  dna_analysis_id uuid,
  CONSTRAINT analysis_history_pkey PRIMARY KEY (id),
  CONSTRAINT analysis_history_google_analysis_id_fkey FOREIGN KEY (google_analysis_id) REFERENCES public.product_analysis_google(id),
  CONSTRAINT analysis_history_perplexity_analysis_id_fkey FOREIGN KEY (perplexity_analysis_id) REFERENCES public.product_analysis_perplexity(id),
  CONSTRAINT analysis_history_dna_analysis_id_fkey FOREIGN KEY (dna_analysis_id) REFERENCES public.product_analysis_dna_google(id),
  CONSTRAINT analysis_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id),
  CONSTRAINT analysis_history_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.analysis_snapshots (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  batch_id uuid,
  product_id uuid NOT NULL,
  status text DEFAULT 'running'::text CHECK (status = ANY (ARRAY['running'::text, 'completed'::text, 'failed'::text, 'partial'::text])),
  started_at timestamp with time zone DEFAULT now(),
  completed_at timestamp with time zone,
  no_of_query integer,
  total_no_of_query integer,
  CONSTRAINT analysis_snapshots_pkey PRIMARY KEY (id),
  CONSTRAINT analysis_snapshots_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES public.query_batches(id),
  CONSTRAINT analysis_snapshots_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.batch_queries (
  batch_id uuid NOT NULL,
  query_id uuid NOT NULL,
  added_at timestamp with time zone DEFAULT now(),
  CONSTRAINT batch_queries_pkey PRIMARY KEY (batch_id, query_id),
  CONSTRAINT batch_queries_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES public.query_batches(id),
  CONSTRAINT batch_queries_query_id_fkey FOREIGN KEY (query_id) REFERENCES public.queries(id)
);
CREATE TABLE public.processed_jobs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  job_id character varying NOT NULL UNIQUE,
  processed_at timestamp with time zone DEFAULT now(),
  expires_at timestamp with time zone DEFAULT (now() + '24:00:00'::interval),
  CONSTRAINT processed_jobs_pkey PRIMARY KEY (id)
);
CREATE TABLE public.product_analysis_dna_google (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  product_id uuid NOT NULL,
  run_id text NOT NULL,
  dna_blueprint jsonb NOT NULL,
  status text DEFAULT 'completed'::text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  input_data_hash character varying,
  CONSTRAINT product_analysis_dna_google_pkey PRIMARY KEY (id),
  CONSTRAINT pa_dna_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.product_analysis_dna_perplexity (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  product_id uuid NOT NULL,
  run_id text NOT NULL,
  dna_blueprint jsonb NOT NULL,
  status text DEFAULT 'completed'::text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  input_data_hash character varying,
  CONSTRAINT product_analysis_dna_perplexity_pkey PRIMARY KEY (id),
  CONSTRAINT pa_dna_perplexity_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.product_analysis_google (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  product_id uuid NOT NULL,
  search_query text NOT NULL,
  google_overview_analysis jsonb,
  raw_serp_results jsonb DEFAULT '[]'::jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  snapshot_id uuid,
  CONSTRAINT product_analysis_google_pkey PRIMARY KEY (id),
  CONSTRAINT product_analysis_google_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.analysis_snapshots(id),
  CONSTRAINT pa_google_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.product_analysis_perplexity (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  product_id uuid NOT NULL,
  optimization_prompt text NOT NULL,
  optimization_analysis jsonb,
  citations jsonb DEFAULT '[]'::jsonb,
  related_google_analysis_id uuid,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  raw_serp_results jsonb,
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  snapshot_id uuid,
  CONSTRAINT product_analysis_perplexity_pkey PRIMARY KEY (id),
  CONSTRAINT product_analysis_perplexity_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.analysis_snapshots(id),
  CONSTRAINT pa_perplexity_google_link_fkey FOREIGN KEY (related_google_analysis_id) REFERENCES public.product_analysis_google(id),
  CONSTRAINT pa_perplexity_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.products (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  user_id uuid NOT NULL,
  product_name text NOT NULL,
  product_url text,
  description text,
  specifications jsonb,
  features jsonb,
  targeted_market text,
  problem_product_is_solving text,
  general_product_type text,
  specific_product_type text,
  generated_query text,
  optimization_analysis jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  source_links jsonb DEFAULT '[]'::jsonb,
  processed_sources jsonb DEFAULT '[]'::jsonb,
  google_overview_analysis jsonb,
  combined_analysis jsonb,
  deep_analysis_google_hash character varying,
  deep_analysis_perplexity_hash character varying,
  CONSTRAINT products_pkey PRIMARY KEY (id),
  CONSTRAINT products_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.user_profiles(id)
);
CREATE TABLE public.queries (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid,
  query_text text NOT NULL,
  priority integer DEFAULT 0,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  product_id uuid,
  google_status USER-DEFINED,
  perplexity_status USER-DEFINED,
  suggested_engine text CHECK (suggested_engine = ANY (ARRAY['google'::text, 'perplexity'::text, 'both'::text])),
  CONSTRAINT queries_pkey PRIMARY KEY (id),
  CONSTRAINT queries_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id),
  CONSTRAINT queries_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id)
);
CREATE TABLE public.query_batches (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  product_id uuid NOT NULL,
  user_id uuid NOT NULL,
  name text NOT NULL,
  description text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT query_batches_pkey PRIMARY KEY (id),
  CONSTRAINT query_batches_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id),
  CONSTRAINT query_batches_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.sov_product_snapshots (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  product_id uuid NOT NULL,
  global_sov_score numeric NOT NULL,
  total_queries_analyzed integer NOT NULL,
  narrative_summary text,
  batch_id text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  engine text NOT NULL DEFAULT 'google'::text,
  citation_score numeric DEFAULT 0,
  category_relevance numeric DEFAULT 0,
  analyzed_at timestamp with time zone DEFAULT now(),
  snapshot_id uuid,
  context_patterns jsonb DEFAULT '{}'::jsonb,
  scraped_generative_dna json,
  CONSTRAINT sov_product_snapshots_pkey PRIMARY KEY (id),
  CONSTRAINT sov_product_snapshots_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.analysis_snapshots(id)
);
CREATE TABLE public.sov_query_insights (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  analysis_id uuid NOT NULL,
  product_id uuid NOT NULL,
  sov_score integer NOT NULL,
  category_relevance text NOT NULL,
  citation_status text NOT NULL,
  winning_source text,
  ai_narrative text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  engine text NOT NULL DEFAULT 'google'::text,
  CONSTRAINT sov_query_insights_pkey PRIMARY KEY (id)
);
CREATE TABLE public.user_profiles (
  id uuid NOT NULL,
  user_name text,
  email text NOT NULL UNIQUE,
  credits integer NOT NULL DEFAULT 100 CHECK (credits >= 0),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  last_login timestamp with time zone,
  avatar_url text,
  subscription_tier text DEFAULT 'free'::text CHECK (subscription_tier = ANY (ARRAY['free'::text, 'pro'::text, 'enterprise'::text])),
  total_products_analyzed integer NOT NULL DEFAULT 0,
  CONSTRAINT user_profiles_pkey PRIMARY KEY (id),
  CONSTRAINT user_profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
