import React from 'react';
import { Banner } from '@neo4j-ndl/react';

export default class ErrorBoundary extends React.Component<any, any> {
  state = { hasError: false, errorMessage: '' };

  static getDerivedStateFromError(_error: unknown) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    this.setState({ ...this.state, errorMessage: error.message });
    console.log({ error });
    console.log({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className='n-size-full n-flex n-flex-col n-items-center n-justify-center n-rounded-md n-bg-palette-neutral-bg-weak n-box-border'>          <Banner
            icon
            type='info'
            description={
              this.state.errorMessage === 'Missing required parameter client_id.'
                ? '请为GCS源提供Google客户端ID'
                : '抱歉，加载此页面时出现问题'
            }
            title='出现了问题'
            floating
            className='mt-8'
            actions={[
              {
                label: '文档',
                href: 'https://github.com/neo4j-labs/llm-graph-builder',
                target: '_blank',
              },
            ]}
          ></Banner>
        </div>
      );
    }
    return this.props.children;
  }
}
